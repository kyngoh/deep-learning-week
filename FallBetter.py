import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import math
from ultralytics import YOLO
from collections import deque

# ============================================================
# MULTI-PERSON "ON-GROUND" FALL DETECTION (Front webcam)
#
# FALL TRIGGER RULE (your requirement):
#   Only trigger FALL if person is ON THE GROUND / FLAT
#   AND holds that position for >= HOLD_SEC (5 seconds).
#
# This avoids squatting false positives.
# ============================================================

# ----------------
# CONFIG
# ----------------
CONF_POSE = 0.5
USE_TRACKING = True
TRACKER_CFG = "bytetrack.yaml"

FPS_EST = 30
HOLD_SEC = 5.0                 # << your requirement (>= 5 sec)

# "On ground" conditions (tuned to avoid squat)
TORSO_FLAT_ANGLE = 75          # very horizontal torso
FLAT_RATIO = 1.35              # bbox width/height > 1.35 indicates lying-ish
VERY_FLAT_RATIO = 1.55         # extra strong lying signal

# Hip close to ankles (collapsed height)
# smaller means hips very near ankles (more like floor posture)
HIP_ANKLE_CLOSE_FRAC = 0.14    # hip_to_ankle < 0.14*h => very low (avoid squat)

# Person is low in the frame (bbox bottom near frame bottom)
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

# ----------------
# HELPERS
# ----------------
def torso_angle_deg(shoulders_mid, hips_mid):
    dx = shoulders_mid[0] - hips_mid[0]
    dy = hips_mid[1] - shoulders_mid[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def init_state():
    return {
        "on_ground_frames": 0,     # how long the person is on ground
        "fallen": False,
        "upright_frames": 0,
        "last_seen_frame": 0,
    }

# ----------------
# MODEL + CAMERA
# ----------------
pose_model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Camera failed to open. Close Zoom/Teams/OBS/Camera app and try again.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1:
    fps = FPS_EST

HOLD_FRAMES = int(HOLD_SEC * fps)
RESET_FRAMES = int(RESET_UPRIGHT_SEC * fps)
MAX_MISSING_FRAMES = int(MISSING_TIMEOUT_SEC * fps)

state = {}
frame_idx = 0

WIN = "Fall Detection (On-ground >= 5s)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

print("Running...")
print("Rule: FALL only if ON-GROUND/FLAT held >= 5 seconds.")
print("Quit: click window then press Q or ESC.")

# ----------------
# MAIN LOOP
# ----------------
while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame.")
        break

    frame_idx += 1
    h, w, _ = frame.shape
    frame_area = h * w

    zx1, zx2 = int(ZONE_X1 * w), int(ZONE_X2 * w)
    if USE_ZONE:
        cv2.rectangle(frame, (zx1, 0), (zx2, h), (255, 255, 255), 1)

    if USE_TRACKING:
        res = pose_model.track(frame, conf=CONF_POSE, verbose=False, persist=True, tracker=TRACKER_CFG)[0]
    else:
        res = pose_model.predict(frame, conf=CONF_POSE, verbose=False)[0]

    drawn = res.plot()
    if USE_ZONE:
        cv2.rectangle(drawn, (zx1, 0), (zx2, h), (255, 255, 255), 1)

    if res.keypoints is None or res.keypoints.xy is None or res.boxes is None:
        # cleanup stale
        stale = [tid for tid, st in state.items() if frame_idx - st["last_seen_frame"] > MAX_MISSING_FRAMES]
        for tid in stale:
            del state[tid]

        cv2.imshow(WIN, drawn)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(10) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        continue

    kpts_list = res.keypoints.xy.cpu().numpy()
    boxes_xyxy = res.boxes.xyxy.cpu().numpy()

    if USE_TRACKING:
        if res.boxes.id is None:
            cv2.putText(drawn, "Tracking IDs missing. Set USE_TRACKING=False.", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow(WIN, drawn)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            continue
        track_ids = res.boxes.id.cpu().numpy().astype(int)
    else:
        track_ids = list(range(len(kpts_list)))

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

        # ---- Strict "on ground / flat" definition ----
        bottom_near_floor = (y2 > (1.0 - BOTTOM_NEAR_FLOOR_FRAC) * h)
        hips_very_close_to_ankles = (hip_to_ankle < HIP_ANKLE_CLOSE_FRAC * h)

        flat_pose = (angle > TORSO_FLAT_ANGLE and ratio > FLAT_RATIO) or (ratio > VERY_FLAT_RATIO)

        # On-ground if:
        # (A) flat_pose AND bottom_near_floor
        # OR
        # (B) hips very close to ankles AND bottom_near_floor
        is_on_ground = (flat_pose and bottom_near_floor) or (hips_very_close_to_ankles and bottom_near_floor)

        # Count how long they are on-ground
        if is_on_ground:
            st["on_ground_frames"] += 1
        else:
            # decay rather than reset instantly (reduces flicker)
            st["on_ground_frames"] = max(0, st["on_ground_frames"] - 2)

        # Trigger FALL only if held >= 5 seconds
        if st["on_ground_frames"] >= HOLD_FRAMES:
            st["fallen"] = True

        # ---- Reset when upright ----
        is_upright = (angle < UPRIGHT_ANGLE) and (hip_to_ankle > (STAND_HEIGHT_FRAC * h)) and (ratio < 1.10)
        if is_upright:
            st["upright_frames"] += 1
        else:
            st["upright_frames"] = 0

        if st["fallen"] and st["upright_frames"] >= RESET_FRAMES:
            st["fallen"] = False
            st["on_ground_frames"] = 0

        # ---- UI ----
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

    # cleanup stale tracks (prevents stuck state)
    stale = [tid for tid, st in state.items() if frame_idx - st["last_seen_frame"] > MAX_MISSING_FRAMES]
    for tid in stale:
        del state[tid]

    cv2.imshow(WIN, drawn)
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break
    key = cv2.waitKey(10) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        break

cap.release()
cv2.destroyAllWindows()
