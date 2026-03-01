import threading
import time
import math
import cv2
import requests
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# allow requests from the dashboard (running on localhost:5173) or any origin during
# development.  the front-end will fetch /video, /incidents and /zone from this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# camera capture globals
# Use DirectShow backend on Windows for more reliable camera access
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] Could not open camera at index 0. Check if another app is using it.")
else:
    print("[OK] Camera opened successfully")

lock = threading.Lock()
latest_frame = None

# simple in-memory store for incidents and zone
incidents: list[dict] = []
zone: str | None = None

# --------------------------------------------------
# fall-detection configuration adapted from FallBetter.py
CONF_POSE = 0.5
USE_TRACKING = True
TRACKER_CFG = "bytetrack.yaml"

FPS_EST = 30
HOLD_SEC = 0.5

# "On ground" conditions (tuned to avoid squat)
TORSO_FLAT_ANGLE = 75          # very horizontal torso
FLAT_RATIO = 1.35              # bbox width/height > 1.35 indicates lying-ish
VERY_FLAT_RATIO = 1.55         # extra strong lying signal

HIP_ANKLE_CLOSE_FRAC = 0.14    # hip_to_ankle < 0.14*h => very low (avoid squat)
BOTTOM_NEAR_FLOOR_FRAC = 0.10  # y2 > (1 - 0.10)*h => bottom 10% of image

# Reset when upright
UPRIGHT_ANGLE = 35
STAND_HEIGHT_FRAC = 0.25
RESET_UPRIGHT_SEC = 1.0

# Light gating
MIN_BOX_AREA_FRAC = 0.01
MISSING_TIMEOUT_SEC = 1.5

# Zone off for webcam
USE_ZONE = False
ZONE_X1, ZONE_X2 = 0.2, 0.8

# --------------------------------------------------
# Telegram Configuration
TELEGRAM_BOT_TOKEN = "8540238802:AAHOWCunUixaMP_SdpZsIKwNgMIrCSUxdkE"
TELEGRAM_CHAT_ID = "365684337"
TELEGRAM_COOLDOWN_SEC = 30


def send_telegram_message(text: str) -> bool:
    """Send a Telegram message using Bot API."""
    if not TELEGRAM_BOT_TOKEN or "PASTE_" in TELEGRAM_BOT_TOKEN:
        print("[Telegram] Bot token not set. Skipping message.")
        return False
    if not TELEGRAM_CHAT_ID or "PASTE_" in str(TELEGRAM_CHAT_ID):
        print("[Telegram] Chat ID not set. Skipping message.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}

    try:
        r = requests.post(url, json=payload, timeout=8)
        if r.status_code != 200:
            print("[Telegram] Failed:", r.status_code, r.text[:200])
            return False
        print("[Telegram] Alert sent successfully")
        return True
    except Exception as e:
        print("[Telegram] Error:", e)
        return False


def torso_angle_deg(shoulders_mid, hips_mid):
    dx = shoulders_mid[0] - hips_mid[0]
    dy = hips_mid[1] - shoulders_mid[1]
    return abs(math.degrees(math.atan2(dx, dy)))


def init_state():
    return {
        "on_ground_frames": 0,
        "fallen": False,
        "upright_frames": 0,
        "last_seen_frame": 0,
        "alert_sent": False,
        "last_alert_time": 0.0,
    }


class Incident(BaseModel):
    timestamp: str
    location: str
    severity: str
    confidence: float
    status: str
    camera: str


# initialize fall detector model + state (runs in same thread as capture)
pose_model = YOLO("yolov8n-pose.pt")
state: dict[int, dict] = {}
frame_idx = 0

# derive timing constants from camera fps
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1:
    fps = FPS_EST
HOLD_FRAMES = int(HOLD_SEC * fps)
RESET_FRAMES = int(RESET_UPRIGHT_SEC * fps)
MAX_MISSING_FRAMES = int(MISSING_TIMEOUT_SEC * fps)


def capture_loop():
    """Continuously read frames, run fall detection, and store annotated output."""
    global latest_frame, frame_idx
    frame_count = 0
    consecutive_failures = 0

    # we keep the capture open for as long as the process lives
    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures % 10 == 0:
                print(f"[WARNING] Failed to read frame (attempt {consecutive_failures}). Error may be -1072873821 (camera busy).")
            # pause briefly and retry if the camera failed to read
            time.sleep(0.1)
            continue
        
        consecutive_failures = 0
        frame_idx += 1
        frame_count += 1
        if frame_count == 1:
            print(f"[OK] First frame captured ({frame.shape})")
        h, w, _ = frame.shape
        frame_area = h * w

        # run pose model (tracking if enabled)
        if USE_TRACKING:
            res = pose_model.track(frame, conf=CONF_POSE, verbose=False, persist=True, tracker=TRACKER_CFG)[0]
        else:
            res = pose_model.predict(frame, conf=CONF_POSE, verbose=False)[0]

        drawn = res.plot()
        if USE_ZONE:
            zx1, zx2 = int(ZONE_X1 * w), int(ZONE_X2 * w)
            cv2.rectangle(drawn, (zx1, 0), (zx2, h), (255, 255, 255), 1)

        # iterate over detections and update state
        if res.keypoints is not None and res.keypoints.xy is not None and res.boxes is not None:
            kpts_list = res.keypoints.xy.cpu().numpy()
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()

            if USE_TRACKING:
                if res.boxes.id is None:
                    track_ids = list(range(len(kpts_list)))
                else:
                    track_ids = res.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = list(range(len(kpts_list)))

            now_time = time.time()
            n = min(len(track_ids), len(kpts_list), len(boxes_xyxy))
            for idx in range(n):
                tid = int(track_ids[idx])
                kpts = kpts_list[idx]
                x1, y1, x2, y2 = boxes_xyxy[idx]
                bw, bh = max(1.0, (x2 - x1)), max(1.0, (y2 - y1))
                box_area = bw * bh

                if len(kpts) < 17:
                    continue
                if box_area < MIN_BOX_AREA_FRAC * frame_area:
                    continue

                if USE_ZONE:
                    ankle_mid_x = float((kpts[15][0] + kpts[16][0]) / 2)
                    if not ((ZONE_X1 * w) < ankle_mid_x < (ZONE_X2 * w)):
                        continue

                if tid not in state:
                    state[tid] = init_state()
                st = state[tid]
                st["last_seen_frame"] = frame_idx

                shoulders_mid = (kpts[5] + kpts[6]) / 2
                hips_mid = (kpts[11] + kpts[12]) / 2
                ankles_mid = (kpts[15] + kpts[16]) / 2

                angle = torso_angle_deg(shoulders_mid, hips_mid)
                hip_to_ankle = abs(float(ankles_mid[1] - hips_mid[1]))
                ratio = float(bw / bh)

                bottom_near_floor = (y2 > (1.0 - BOTTOM_NEAR_FLOOR_FRAC) * h)
                hips_very_close_to_ankles = (hip_to_ankle < HIP_ANKLE_CLOSE_FRAC * h)
                flat_pose = (angle > TORSO_FLAT_ANGLE and ratio > FLAT_RATIO) or (ratio > VERY_FLAT_RATIO)
                is_on_ground = (flat_pose and bottom_near_floor) or (hips_very_close_to_ankles and bottom_near_floor)

                if is_on_ground:
                    st["on_ground_frames"] += 1
                else:
                    st["on_ground_frames"] = max(0, st["on_ground_frames"] - 2)

                # debug logging every 50 frames to show what's happening
                if frame_idx % 50 == 0:
                    print(f"[debug] tid={tid} angle={angle:.1f}° ratio={ratio:.2f} "
                          f"on_ground={st['on_ground_frames']} frames (~{st['on_ground_frames']/fps:.2f}s) "
                          f"is_on_ground={is_on_ground} bottom_near_floor={bottom_near_floor}")

                was_fallen = st["fallen"]
                if st["on_ground_frames"] >= HOLD_FRAMES:
                    st["fallen"] = True

                is_upright = (angle < UPRIGHT_ANGLE) and (hip_to_ankle > (STAND_HEIGHT_FRAC * h)) and (ratio < 1.10)
                if is_upright:
                    st["upright_frames"] += 1
                else:
                    st["upright_frames"] = 0

                if st["fallen"] and st["upright_frames"] >= RESET_FRAMES:
                    st["fallen"] = False
                    st["on_ground_frames"] = 0
                    st["alert_sent"] = False  # allow new alert next time
                    # do NOT reset last_alert_time (cooldown still applies)

                just_fell = (not was_fallen) and st["fallen"]
                if just_fell:
                    print(f"[FALL] tid={tid} detected! was on ground for {st['on_ground_frames']} frames (~{st['on_ground_frames']/fps:.1f}s)")
                    # record incident
                    with lock:
                        incidents.append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "location": zone or "unknown",
                            "severity": "high",
                            "confidence": 1.0,
                            "status": "pending",
                            "camera": "webcam",
                        })

                    # Telegram alert with cooldown
                    if not st["alert_sent"]:
                        if now_time - st["last_alert_time"] >= TELEGRAM_COOLDOWN_SEC:
                            seconds_down = st["on_ground_frames"] / fps
                            msg = f"🚨 FALL DETECTED (ID:{tid})\\nDown for ~{seconds_down:.1f}s.\\nPlease check immediately."
                            sent = send_telegram_message(msg)
                            st["last_alert_time"] = now_time
                            st["alert_sent"] = True
                        else:
                            st["alert_sent"] = True

                # drawing overlays
                tx, ty = int(x1), int(max(0, y1 - 10))
                cv2.putText(drawn, f"ID:{tid}", (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(drawn, f"ratio:{ratio:.2f} ang:{angle:.0f}", (tx, ty + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(drawn, f"ground:{st['on_ground_frames']/fps:.1f}s floor:{int(bottom_near_floor)}",
                            (tx, ty + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                if st["fallen"]:
                    cv2.putText(drawn, "FALL DETECTED", (tx, ty + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # cleanup stale tracks
            stale = [tid for tid, st in state.items() if frame_idx - st["last_seen_frame"] > MAX_MISSING_FRAMES]
            for tid in stale:
                del state[tid]

        # update frame for streaming
        with lock:
            latest_frame = drawn.copy()

        # throttle the loop to avoid hogging the CPU
        time.sleep(0.03)


# start background thread for grabbing frames
print("[...] Starting camera capture thread...")
threading.Thread(target=capture_loop, daemon=True).start()
# give it a moment to grab the first frame
time.sleep(1)
if latest_frame is None:
    print("[WARNING] No frames captured after 1 second. Camera may not be working.")
else:
    print(f"[OK] Frames being captured. API at http://localhost:8000")


def _frame_generator():
    """Yield multipart JPEG frames suitable for an <img> tag pointing at /video."""
    frame_count = 0
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            # silently wait without spamming logs
            time.sleep(0.1)
            continue

        frame_count += 1
        # encode to jpeg
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        data = jpeg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.03)


@app.get("/video")
def video_feed():
    """Live video stream endpoint that the dashboard can consume."""
    return StreamingResponse(_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/incidents")
def get_incidents():
    """Return the current list of detected incidents."""
    return JSONResponse(content=incidents)


@app.put("/zone")
def set_zone(payload: dict):
    """Receive the selected zone from the dashboard."""
    global zone
    zone = payload.get("zone")
    return {}


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI server—this blocks and keeps the process alive
    uvicorn.run("edgefall_api:app", host="0.0.0.0", port=8000)