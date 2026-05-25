# Interview Guide

## Two-minute opening

This project solves anomaly detection over server telemetry with a temporal binary classifier in PyTorch. The core ML pipeline uses a chronological split, a scaler fitted only on training data, and a causal 5-step sliding window to avoid leakage. I first delivered a robust MLP path, then added a GRU variant without breaking the original delivery path.

On top of the trained model, I built a local FastAPI demo with three layers of value:

1. Snapshot inference with sliders for quick qualitative exploration.
2. Batch CSV analysis with anomaly summaries, downloadable scored output and a temporal chart.
3. A documentation-grounded RAG assistant that answers project questions with citations from the repo docs.

The emphasis was not just “getting metrics”, but making the whole system reproducible, explainable and demoable.

## Technical highlights to mention

* Training/evaluation reuse the persisted scaler instead of silently refitting it.
* The model checkpoint stores the real training configuration.
* The GRU path reuses shared training utilities rather than duplicating loop logic.
* The web app is honest about snapshot inference: it repeats one reading 5 times because the model expects temporal windows.
* Batch inference is more faithful because it scores a real sequential series.
* The RAG pipeline is grounded on repository documentation and exposes citations in both CLI and web.

## If they ask “why MLP first?”

Because the MLP was the minimum reliable deliverable. It solved the task with a clean temporal pipeline and strong calibrated results. The GRU was added afterwards as an extension, but I kept the first path stable so the project remained deliverable even if the extension had issues.

## If they ask “what would you harden for production?”

* Add authentication and request limits to FastAPI.
* Move long-running CSV analysis to background jobs.
* Persist sessions in object storage or a database instead of local files.
* Replace permissive FAISS deserialization with a safer persistence strategy.
* Add proper backend integration tests with the full web stack installed.
* Add observability around inference latency and model/drift monitoring.

## If they ask “what are the current limitations?”

* Snapshot slider mode is approximate because the model was trained on temporal sequences.
* Batch mode uses the training scaler, so out-of-distribution uploads are clipped and warned about rather than recalibrated online.
* The RAG service depends on Ollama availability unless a local fallback model is present.

## Short answers for predictable questions

### Why not use random shuffle?
Because this is time-series telemetry. Shuffling would create look-ahead leakage and inflate performance.

### Why use `BCEWithLogitsLoss`?
It is numerically more stable than applying sigmoid first and then binary cross-entropy separately.

### Why tune the threshold?
The default 0.5 was not optimal for the class imbalance and precision/recall tradeoff. Validation-based threshold calibration materially improved F1.

### Why does the web demo matter?
Because it converts an ML notebook-style result into something inspectable, reproducible and usable by non-ML reviewers during the technical assessment.
