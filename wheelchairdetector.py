import os
import cv2
import math
import time
from ultralytics import YOLO, YOLOWorld
from collections import deque

# --- Environment Setup (Mac Specific) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ============================================================
# 1) LOAD MODELS (YOLO-World & YOLOv8-Pose)
# ============================================================
pose_model = YOLO("yolov8n-pose.pt")

try:
    obj_model = YOLOWorld("yolov8s-worldv2.pt")
    # Natural Language classes for mobility aids
    VULNERABLE_CLASSES = [
        "walking stick", "cane", "crutch", "wheelchair", 
        "walker", "dustpan", "broom", "chair", "umbrella", "baseball bat"
    ]
    obj_model.set_classes(VULNERABLE_CLASSES)
    print("YOLO-World initialized successfully.")
except Exception as e:
    print(f"YOLO-World failed: {e}. Falling back to standard YOLOv8n.")
    obj_model = YOLO("yolov8n.pt")

# ============================================================
# 2) CONFIG & CAMERA (Mac Fix)
# ============================================================
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise RuntimeError("Camera failed to open. Check Mac Privacy Settings.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1:
    fps = 30.0

# Detection Zones & Floor Logic
ZONE_X1, ZONE_X2 = 0.2, 0.8  # Safety zone (middle 60%)
NEAR_FLOOR_FRAC = 0.30       # Bottom 30% of image is the "ground area"

# Thresholds
HUNCH_ANGLE_THRESHOLD = 20   # Degrees lean for elderly detection
GAIT_THRESHOLD = 15          # Pixel movement for slow walking
HOLD_SEC = 2.5               # Shortened for demo; person must stay down for 2.5s
HOLD_FRAMES = int(HOLD_SEC * fps)

# Fall Geometry Constants
TORSO_FLAT_ANGLE = 75        # Horizontal torso
FLAT_RATIO = 1.35            # BBox width/height ratio
HIP_ANKLE_CLOSE_FRAC = 0.16  # Vertical distance between hip and ankle
FRAME_WINDOW = 10

# -----------------------------
# 3) HELPERS + STATE
# -----------------------------
def torso_angle_deg(shoulders_mid, hips_mid):
    dx = shoulders_mid[0] - hips_mid[0]
    dy = hips_mid[1] - shoulders_mid[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def init_person_state():
    return {
        "on_ground_frames": 0,
        "fallen": False,
        "last_seen_frame": 0,
        "history_angle": deque(maxlen=5),
        "history_velocity": deque(maxlen=FRAME_WINDOW)
    }

person_states = {}
frame_idx = 0
WIN_NAME = "LTA Smart Guardian Pro"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

# -----------------------------
# 4) MAIN LOOP
# -----------------------------
while True:
    ok, frame = cap.read()
    if not ok: break

    frame_idx += 1
    h, w, _ = frame.shape
    trigger_extension = False
    floor_y = int((1.0 - NEAR_FLOOR_FRAC) * h)

    # --- STEP A: Object Triggers (Mobility Aids) ---
    obj_results = obj_model.predict(frame, conf=0.25, verbose=False)[0]
    if obj_results.boxes is not None:
        for box in obj_results.boxes:
            label = obj_model.names[int(box.cls[0])]
            trigger_extension = True 
            b = box.xyxy[0].cpu().numpy()
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)
            cv2.putText(frame, f"AID: {label}", (int(b[0]), max(0, int(b[1]) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- STEP B: Pose Tracking (Fall & Biometrics) ---
    pose_res = pose_model.track(frame, conf=0.5, verbose=False, persist=True, tracker="bytetrack.yaml")[0]
    frame = pose_res.plot()

    if pose_res.keypoints is not None and pose_res.boxes.id is not None:
        kpts_list = pose_res.keypoints.xy.cpu().numpy()
        boxes_xyxy = pose_res.boxes.xyxy.cpu().numpy()
        track_ids = pose_res.boxes.id.cpu().numpy().astype(int)

        for idx in range(len(track_ids)):
            tid = int(track_ids[idx])
            kpts = kpts_list[idx]
            x1, y1, x2, y2 = boxes_xyxy[idx]
            
            if len(kpts) < 17: continue
            if tid not in person_states: person_states[tid] = init_person_state()
            
            st = person_states[tid]
            st["last_seen_frame"] = frame_idx

            # Centers for analysis
            shoulders = (kpts[5] + kpts[6]) / 2
            hips = (kpts[11] + kpts[12]) / 2
            ankles = (kpts[15] + kpts[16]) / 2

            # 1. Hunch & Gait Logic
            angle = torso_angle_deg(shoulders, hips)
            st["history_angle"].append(angle)
            avg_angle = sum(st["history_angle"]) / len(st["history_angle"])

            st["history_velocity"].append(float(ankles[0]))
            velocity = abs(st["history_velocity"][-1] - st["history_velocity"][0]) if len(st["history_velocity"]) == FRAME_WINDOW else 20

            if (ZONE_X1 * w < ankles[0] < ZONE_X2 * w):
                if avg_angle > HUNCH_ANGLE_THRESHOLD or velocity < GAIT_THRESHOLD:
                    trigger_extension = True

            # 2. IMPROVED FALL DETECTION LOGIC
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            ratio = bw / bh
            hip_to_ankle_v = abs(float(ankles[1] - hips[1]))
            
            # Checks: horizontal pose OR collapsed height near ground
            is_flat = (angle > TORSO_FLAT_ANGLE) or (ratio > 1.5)
            is_collapsed = (hip_to_ankle_v < HIP_ANKLE_CLOSE_FRAC * h)
            near_ground = (hips[1] >= floor_y) or (ankles[1] >= floor_y) or (y2 >= floor_y)

            if near_ground and (is_flat or is_collapsed):
                st["on_ground_frames"] += 1
            else:
                st["on_ground_frames"] = max(0, st["on_ground_frames"] - 1) # Slower decay for reliability

            if st["on_ground_frames"] >= HOLD_FRAMES:
                st["fallen"] = True
                trigger_extension = True

            # --- UI FEEDBACK ---
            if st["fallen"]:
                cv2.rectangle(frame, (int(x1), int(y1) - 60), (int(x1) + 350, int(y1)), (0, 0, 255), -1)
                cv2.putText(frame, "FALLEN", (int(x1) + 10, int(y1) - 15), 
                            cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 4)

    # --- STEP C: SYSTEM HUD ---
    cv2.rectangle(frame, (int(ZONE_X1*w), 0), (int(ZONE_X2*w), h), (255, 255, 255), 1) # Zone
    cv2.line(frame, (0, floor_y), (w, floor_y), (0, 255, 255), 1) # Floor Line
    
    status_text = "EXTEND GREEN MAN" if trigger_extension else "NORMAL SIGNAL"
    s_color = (0, 0, 255) if trigger_extension else (0, 255, 0)
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (w//4, 45), 0, 1.2, s_color, 3)

    cv2.imshow(WIN_NAME, frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break

cap.release()
cv2.destroyAllWindows()
