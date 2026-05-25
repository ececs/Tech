# Demo Checklist

## Before the demo

1. Activate the environment and confirm dependencies:
```bash
source .venv/bin/activate
python3 -V
uv run pytest tests/test_model.py
```

2. Confirm the trained artefacts exist:
```bash
ls best_model.pth artifacts/scaler.joblib
```

3. If you want the RAG chat available, verify Ollama:
```bash
curl -s "${OLLAMA_URL:-http://192.168.31.181:11434}/api/tags"
```

The project defaults to the secondary RTX 4070 Super workstation. If that host is unreachable, the CLI and app will try `tinyllama` locally (override with `--ollama-url`).

## Recommended demo flow

1. Show the project architecture briefly:
   Explain the temporal split, scaler persistence, sliding window and calibrated threshold.

2. Show model evaluation quality:
```bash
make evaluate
```

3. Launch the dashboard:
```bash
make app
```

4. In the UI, show snapshot inference first:
   Move CPU temp and memory sliders upward until the anomaly probability rises.

5. Upload a CSV:
   Show the summary cards, anomaly rate and temporal chart.

6. Download analyzed results:
   Mention that the exported CSV adds `probability` and `prediction`.

7. Show the documentation chat:
   Example questions:
   * `¿Qué threshold final se usa?`
   * `¿Por qué el GRU pierde frente al MLP?`
   * `¿Cómo se calcula el pos_weight?`

## Key talking points

* The MLP path is fully deliverable on its own.
* The GRU path was added afterwards without breaking the original delivery path.
* Inference and evaluation reuse the persisted scaler, avoiding leakage/regeneration bugs.
* The FastAPI demo is honest about snapshot inference: it repeats the same reading 5 times because the model was trained on temporal windows.
* Batch CSV mode is more representative because it preserves sequential structure.
* The RAG assistant is grounded on project documentation and returns citations.

## Fallback notes

* If FastAPI dependencies are missing:
```bash
python3 -m pip install -r requirements.txt
```

* If Ollama is unreachable:
  Explain that the system degrades gracefully and returns a clear error or uses local fallback when available.

* If asked about production hardening:
  Mention that authentication, background workers, request limits and persisted application state were intentionally left out because this is a local technical demo.
