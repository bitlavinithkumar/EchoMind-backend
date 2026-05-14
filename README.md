# Backend Setup & Run

## Folder to place this: anywhere you like, e.g. ~/chatbot/backend/

## 1. Install Python dependencies
```bash
cd backend/
pip install -r requirements.txt
```

## 2. Configure environment
Edit `.env` — update MONGODB_URI, GEMINI_API_KEY etc. if needed.
FRONTEND_ORIGIN must match the URL your frontend runs on (default: http://localhost:5174).

## 3. Run
```bash
uvicorn main_hybrid:app --reload --port 8000
```

The API will be live at http://localhost:8000
Swagger docs: http://localhost:8000/docs
