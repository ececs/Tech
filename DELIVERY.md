# Delivery Guide

## What to include

For the safest delivery, include the full project with these artefacts already present:

* `best_model.pth`
* `artifacts/scaler.joblib`
* `artifacts/rag_index/` if you want the RAG chat to work immediately without rebuilding the FAISS index

Recommended delivery options:

1. Full repository as a `.zip`
2. Git repository plus artefacts if the evaluator accepts binary outputs

## Fastest evaluator path

If the evaluator wants to see the web app quickly:

### Option A: Local Python
```bash
pip install -r requirements.txt
uvicorn src.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

### Option B: Docker
```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000
```

## If artefacts are missing

The evaluator must generate the trained artefacts first:

```bash
python -m src.train \
  --epochs 50 \
  --patience 7 \
  --batch-size 64 \
  --hidden-dims 64 32 \
  --lr 3e-3 \
  --dropout 0.2 \
  --pos-weight-scale 0.25 \
  --decision-threshold 0.71
```

After that, the web app can be launched normally.

## Notes for the evaluator

* Snapshot slider mode is intentionally approximate because the model was trained on temporal windows.
* Batch CSV mode is more representative of the actual training setup.
* The RAG chat depends on Ollama availability. If the remote server is unreachable, the app may fall back to `tinyllama` locally if available.

## Suggested files to read first

* [README.md](README.md)
* [DEMO_CHECKLIST.md](DEMO_CHECKLIST.md)
* [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md)
