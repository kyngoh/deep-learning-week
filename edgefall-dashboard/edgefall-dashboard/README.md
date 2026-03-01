# EdgeFall SG Dashboard

React + Tailwind CSS dashboard for the EdgeFall SG Community Safety Command Center.

## Prerequisites

- Node.js 18+ and npm
- Python backend running (`edgefall_api.py` in the parent folder)

## Setup

1. Install dependencies:

```bash
npm install
```

2. Start the Python API (from the project root):

```bash
cd ..
pip install fastapi uvicorn  # if not already installed
python edgefall_api.py
```

3. Start the dashboard:

```bash
npm run dev
```

4. Open http://localhost:5173 in your browser.

## Build for production

```bash
npm run build
npm run preview
```
