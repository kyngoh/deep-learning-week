# EdgeFall Deep Learning Workspace

This repository contains a fall‑detection system built around a YOLOv8 pose model and a React dashboard. The Python backend captures a webcam feed, performs fall detection using pose estimation, and streams video/alerts to a frontend dashboard. Optional Telegram notifications are supported for incident alerts.

---

## 📁 Repository Structure

```
deep-learning-week/
├─ edgefall_api.py            # FastAPI backend + fall detection logic
├─ FallBetter.py              # original standalone fall detection script
├─ dlw.py                     # miscellaneous script
├─ requirements.txt           # Python dependencies
└─ edgefall-dashboard/        # React + Tailwind frontend app
    └─ edgefall-dashboard/
        ├─ package.json
        ├─ src/
        └─ ...
```

---

## 🛠️ Prerequisites

- **Operating System:** Windows (camera backend uses DirectShow)
- **Python:** 3.11+ (virtual environment strongly recommended)
- **Node.js:** 18+ and npm for the dashboard
- **Git:** to clone/pull the repo

---

## ⚙️ Python Environment Setup

1. **Create & activate a virtualenv**
   ```powershell
   cd C:\\Users\\kynas\\Downloads\\deep-learning-week
   python -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1
   ```

2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Camera permissions**
   - Make sure no other application (Zoom, Teams, OBS) is using the webcam.
   - EdgeFall uses OpenCV with the `CAP_DSHOW` backend on Windows.

---

## 🚀 Running the Backend

From the workspace root:

```powershell
python edgefall_api.py
```

You should see startup messages like:
```
[OK] Camera opened successfully
[...] Starting camera capture thread...
[OK] First frame captured (480, 640, 3)
[OK] Frames being captured. API at http://localhost:8000
INFO:     Uvicorn running on http://0.0.0.0:8000
```

- **Live video stream:** `GET http://localhost:8000/video` (MJPEG)
- **Incident log:** `GET http://localhost:8000/incidents`
- **Set zone:** `PUT http://localhost:8000/zone` with JSON `{"zone":"<name>"}`

The backend performs fall detection on each frame and adds incidents when a person is on the ground for ≥ 5 seconds. It also sends Telegram alerts if the bot is configured.

---

## 📦 Frontend Dashboard

1. Navigate to the dashboard folder:
   ```powershell
   cd edgefall-dashboard/edgefall-dashboard
   ```

2. Install packages:
   ```powershell
   npm install
   ```

3. Run development server:
   ```powershell
   npm run dev
   ```

4. Open `http://localhost:5173` in your browser.

The dashboard will automatically pull the live camera feed and incident list from the backend.

---

## ✨ Telemetry & Notifications

- **Telegram alerts** can be enabled by editing the constants in `edgefall_api.py`:
  ```python
  TELEGRAM_BOT_TOKEN = "<your-token>"
  TELEGRAM_CHAT_ID = "<your-chat-id>"
  TELEGRAM_COOLDOWN_SEC = 30
  ```
- When a fall is detected, the backend attempts to send a message and logs success/failure.

---

## 🧠 Training / Model Code

Source files under `src/` implement data preprocessing and training logic (`train.py`, `model.py`, `run_realtime.py`, etc.) but are outside the scope of the live demo. You can use them to retrain or evaluate the fall model.

---

## 📄 Notes

- Existing original script `FallBetter.py` remains for reference.
- The API and dashboard communicate over localhost; CORS is configured for ease of development.
- Adjust fall‑detection thresholds in `edgefall_api.py` if your camera angle or environment differs.

---

## ✅ Summary

1. Activate Python venv and install requirements.
2. Start `edgefall_api.py` for backend + detection.
3. Launch the React dashboard with `npm run dev`.
4. (Optional) Configure Telegram to receive alerts.

Enjoy testing your live fall detection system!
