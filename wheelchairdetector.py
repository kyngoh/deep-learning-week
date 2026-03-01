import os
import cv2
import math
import time
from ultralytics import YOLO, YOLOWorld
from collections import deque

# --- Environment Setup ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ============================================================
# 1) LOAD MODELS (YOLO-World for LVIS-scale & YOLOv8-Pose)
# ============================================================
pose_model = YOLO("yolov8n-pose.pt")

# Using YOLO-World fixes the FileNotFoundError for LVIS weights
try:
    obj_model = YOLOWorld("yolov8s-worldv2.pt")
    # Define the 'Natural Language' classes to detect
    VULNERABLE_CLASSES = ["walking stick", "cane", "crutch", "wheelchair", "walker", "dustpan", "broom", "chair", "umbrella", "baseball bat"]
    obj_model.set_classes(VULNERABLE_CLASSES + ["chair", "umbrella"])
    USING_WORLD = True
    print("YOLO-World initialized with LVIS-scale vocabulary.")
except Exception as e:
    print(f"YOLO-World failed to load: {e}. Falling back to standard COCO.")
    obj_model = YOLO("yolov8n.pt")
    USING_WORLD = False

# ============================================================
# 2) CONFIG & CAMERA (Mac-Specific Fix)
# ============================================================
# On Mac, we remove CAP_DSHOW and use default index 0
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise RuntimeError("Camera failed to open. Ensure no other app (Zoom/Teams) is using it.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1:
    fps = 30.0

# Detection Zones
ZONE_X1, ZONE_X2 = 0.2, 0.8  # Safety zone (middle 60% of road)
NEAR_FLOOR_FRAC = 0.22       # Bottom 22% of image is the "ground"

# Thresholds
HUNCH_ANGLE_THRESHOLD = 20   # Degrees lean
GAIT_THRESHOLD = 15          # Pixel movement over 10 frames
HOLD_SEC = 5.0               # Fall must be held for 5 seconds
HOLD_FRAMES = int(HOLD_SEC * fps)

# Fall Geometry
TORSO_FLAT_ANGLE = 75
FLAT_RATIO = 1.35
HIP_ANKLE_CLOSE_FRAC = 0.16 
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
WIN_NAME = "LTA Smart Guardian Pro (Unified)"
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

    # --- STEP A: Object Triggers (YOLO-World) ---
    obj_results = obj_model.predict(frame, conf=0.25, verbose=False)[0]
    if obj_results.boxes is not None:
        for box in obj_results.boxes:
            label = obj_model.names[int(box.cls[0])]
            trigger_extension = True # If any set class is detected
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

            # Centers
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

            # 2. Fall Detection Logic
            ratio = (x2 - x1) / max(1, (y2 - y1))
            hip_to_ankle = abs(float(ankles[1] - hips[1]))
            
            is_on_ground = (hips[1] >= floor_y) and ((angle > TORSO_FLAT_ANGLE and ratio > FLAT_RATIO) or (hip_to_ankle < HIP_ANKLE_CLOSE_FRAC * h))
            
            if is_on_ground:
                st["on_ground_frames"] += 1
            else:
                st["on_ground_frames"] = max(0, st["on_ground_frames"] - 2)

            if st["on_ground_frames"] >= HOLD_FRAMES:
                st["fallen"] = True
                trigger_extension = True # Fall in road must hold traffic

            # --- UPDATED UI FOR HIGH VISIBILITY ---
            if st["fallen"]:
                # Draw a thick red background box behind the text for maximum contrast
                cv2.rectangle(frame, (int(x1), int(y1) - 60), (int(x1) + 400, int(y1)), (0, 0, 255), -1)
                
                # Large, Bold White Text on Red Background
                cv2.putText(frame, "FALLEN", (int(x1) + 10, int(y1) - 15), 
                            cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 5)
                
                trigger_extension = True # Fall in road must hold traffic

    # --- STEP C: HUD & DISPLAY ---
    # Draw Safety Zone
    cv2.rectangle(frame, (int(ZONE_X1*w), 0), (int(ZONE_X2*w), h), (255, 255, 255), 1)
    
    # Status Bar
    status_text = "EXTEND GREEN MAN" if trigger_extension else "NORMAL SIGNAL"
    s_color = (0, 0, 255) if trigger_extension else (0, 255, 0)
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (w//4, 45), 0, 1.2, s_color, 3)

    cv2.imshow(WIN_NAME, frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break

cap.release()
cv2.destroyAllWindows()
