# Installation Guide

Use these commands from `D:\Dymeter\METER-main`.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd frontend
npm.cmd install
```

Run the backend:

```powershell
uvicorn researchApp.api.server:app --reload --host 127.0.0.1 --port 8000
```

Run the frontend:

```powershell
cd frontend
npm.cmd run dev
```

Open the local Vite URL, usually `http://127.0.0.1:5173`.

For the grounded chat panel, set `ANTHROPIC_API_KEY` before starting the backend. If the key is not set, the app uses a local deterministic answer that only reads the saved run JSON.

