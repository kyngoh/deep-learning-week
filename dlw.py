import os
import csv
import cv2
import winsound
import time
import numpy as np
from collections import deque
from ultralytics import YOLO

# -----------------------------
# CONFIG (tune these)
# -----------------------------
MODEL_YOLO = "yolov8n.pt"
CONF_THRES = 0.35
IOU_THRES = 0.45

# Temporal settings
WINDOW = 20                 # frames to keep (~0.7s at 30fps)
DROP_PIXELS_THRES = 25      # sudden downward shift threshold (pixels per frame)
HORIZONTAL_AR_THRES = 1.25  # width/height > this suggests lying/horizontal
SUSTAIN_FRAMES = 10         # how many frames horizontal to confirm fall
ALERT_COOLDOWN = 3.0        # seconds between alerts for same person

# Optional Crosswalk ROI (x1, y1, x2, y2) in pixels. Set None to disable.
CROSSWALK_ROI = None  # e.g. (200, 300, 1100, 900)

# Evidence saving

EVIDENCE_DIR = r"C:\Users\trini\Downloads\falls"  # folder will be created automatically
SAVE_CROPPED_PERSON = True  # also save the cropped person image


# -----------------------------
# Utility
# -----------------------------
def play_alarm():
    winsound.Beep(1200, 300)
    winsound.Beep(1200, 300)
    winsound.Beep(1200, 300)


def box_in_roi(box, roi):
    if roi is None:
        return True
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)


def aspect_ratio(box):
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return w / h


def centroid(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


# -----------------------------
# Simple centroid tracker
# -----------------------------
class CentroidTracker:
    def __init__(self, max_lost=15, dist_thresh=80):
        self.next_id = 0
        self.objects = {}  # id -> (cx, cy)
        self.lost = {}     # id -> lost frames
        self.max_lost = max_lost
        self.dist_thresh = dist_thresh

    def update(self, detections):
        """
        detections: list of boxes (x1,y1,x2,y2)
        returns dict: id -> box
        """
        if len(detections) == 0:
            for oid in list(self.lost.keys()):
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.objects.pop(oid, None)
                    self.lost.pop(oid, None)
            return {}

        det_centroids = [centroid(b) for b in detections]

        if len(self.objects) == 0:
            assigned = {}
            for i, box in enumerate(detections):
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = det_centroids[i]
                self.lost[oid] = 0
                assigned[oid] = box
            return assigned

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self.objects[oid] for oid in obj_ids], dtype=np.float32)
        det_centroids_np = np.array(det_centroids, dtype=np.float32)

        dists = np.linalg.norm(obj_centroids[:, None, :] - det_centroids_np[None, :, :], axis=2)

        used_dets = set()
        assigned = {}

        # Greedy match: smallest distance first
        for r in np.argsort(dists.min(axis=1)):
            oid = obj_ids[r]
            c = int(np.argmin(dists[r]))
            if c in used_dets:
                continue
            if dists[r, c] > self.dist_thresh:
                continue
            used_dets.add(c)
            self.objects[oid] = tuple(det_centroids[c])
            self.lost[oid] = 0
            assigned[oid] = detections[c]

        # New detections get new IDs
        for i, box in enumerate(detections):
            if i in used_dets:
                continue
            oid = self.next_id
            self.next_id += 1
            self.objects[oid] = tuple(det_centroids[i])
            self.lost[oid] = 0
            assigned[oid] = box

        # Mark unmatched old objects as lost
        assigned_ids = set(assigned.keys())
        for oid in list(self.objects.keys()):
            if oid not in assigned_ids:
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.objects.pop(oid, None)
                    self.lost.pop(oid, None)

        return assigned


# -----------------------------
# Fall logic per tracked ID
# -----------------------------
class FallState:
    def __init__(self):
        self.cy_hist = deque(maxlen=WINDOW)   # centroid y history
        self.ar_hist = deque(maxlen=WINDOW)   # aspect ratio history
        self.horizontal_count = 0
        self.last_alert_time = 0.0

    def update(self, cy, ar):
        self.cy_hist.append(cy)
        self.ar_hist.append(ar)

        drop_flag = False
        if len(self.cy_hist) >= 2:
            dy = self.cy_hist[-1] - self.cy_hist[-2]
            if dy >= DROP_PIXELS_THRES:
                drop_flag = True

        horizontal_flag = ar >= HORIZONTAL_AR_THRES
        if horizontal_flag:
            self.horizontal_count += 1
        else:
            self.horizontal_count = max(0, self.horizontal_count - 1)

        sustained_horizontal = self.horizontal_count >= SUSTAIN_FRAMES

        # Trigger: sudden drop + sustained horizontal posture
        fall_now = drop_flag and sustained_horizontal
        return drop_flag, horizontal_flag, sustained_horizontal, fall_now


# -----------------------------
# Main
# -----------------------------
def main(video_source=0):
    yolo = YOLO(MODEL_YOLO)
    tracker = CentroidTracker()
    states = {}

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video source")

    # ----- Evidence saving setup -----
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    csv_path = os.path.join(EVIDENCE_DIR, "fall_log.csv")
    csv_exists = os.path.exists(csv_path)

    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["timestamp", "person_id", "x1", "y1", "x2", "y2", "evidence_image", "crop_image"])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # ROI overlay
            if CROSSWALK_ROI is not None:
                rx1, ry1, rx2, ry2 = CROSSWALK_ROI
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
                cv2.putText(frame, "CROSSWALK ROI", (rx1, max(0, ry1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # YOLO detect persons
            results = yolo.predict(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
            dets = []
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls = int(b.cls[0].item())
                    if cls != 0:  # person
                        continue
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                    box = (x1, y1, x2, y2)
                    if box_in_roi(box, CROSSWALK_ROI):
                        dets.append(box)

            tracked = tracker.update(dets)

            # Process tracked people
            for pid, box in tracked.items():
                x1, y1, x2, y2 = clamp_box(box, w, h)
                box = (x1, y1, x2, y2)

                cx, cy = centroid(box)
                ar = aspect_ratio(box)

                if pid not in states:
                    states[pid] = FallState()

                drop_flag, horiz_flag, sustain_flag, fall_now = states[pid].update(cy, ar)

                now = time.time()
                alert = False
                if fall_now and (now - states[pid].last_alert_time > ALERT_COOLDOWN):
                    alert = True
                    states[pid].last_alert_time = now

                # draw box
                color = (0, 0, 255) if alert else (255, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"ID {pid} | AR={ar:.2f}"
                cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                tags = []
                if drop_flag: tags.append("DROP")
                if horiz_flag: tags.append("HORIZONTAL")
                if sustain_flag: tags.append("SUSTAINED")
                if alert: tags.append("FALL ALERT")

                if tags:
                    cv2.putText(frame, " ".join(tags), (x1, min(h - 10, y2 + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                # ----- ALERT ACTIONS -----
                if alert:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    ts_file = time.strftime("%Y%m%d_%H%M%S")

                    print(f"[{ts}] [ALERT] Fall-like event: ID={pid}, box={box}")
                    play_alarm()

                    # Save evidence image (full frame with boxes/tags)
                    evidence_name = f"fall_{ts_file}_id{pid}.jpg"
                    evidence_path = os.path.join(EVIDENCE_DIR, evidence_name)
                    cv2.imwrite(evidence_path, frame)

                    # Save cropped person (optional)
                    crop_name = ""
                    if SAVE_CROPPED_PERSON:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop_name = f"fall_crop_{ts_file}_id{pid}.jpg"
                            crop_path = os.path.join(EVIDENCE_DIR, crop_name)
                            cv2.imwrite(crop_path, crop)

                    # Log to CSV
                    csv_writer.writerow([ts, pid, x1, y1, x2, y2, evidence_name, crop_name])
                    csv_file.flush()

                    print(f"Saved evidence -> {evidence_path}")

            cv2.imshow("DLW Traffic Light Fall Detection (BBox Baseline)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 0 = webcam. Or use "yourvideo.mp4"
    main(video_source=0)
