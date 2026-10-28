# AI Manufacturing Optimization — VS Code Starter

This is a React + FastAPI starter project for your manufacturing dashboard.

**Important:** The actual Kaggle CSVs and trained model are not included because they are not available in this chat. Copy your downloaded Kaggle bundle's `phase4_outputs` CSVs into `backend/data/phase4_outputs/`. Optional cleaned CSVs go into `backend/data/cleaned_dataset/`.

## Run backend (Windows PowerShell)
```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

## Run frontend (open a second terminal)
Install Node.js LTS, then:
```powershell
cd frontend
npm install
npm run dev
```
Open the Vite URL shown in the terminal (usually http://localhost:5173).

The dashboard displays exported results. It does not retrain the model or create new predictions yet.
