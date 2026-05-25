# TechStream Anomaly Detection System

This repository contains the complete implementation of a Deep Learning system in PyTorch for detecting anomalies in server telemetry data. The system is designed to predict system failures caused by thermal stress and memory overload.

---

## 🎯 Project Overview & Core Architecture

The system follows a modular, production-ready machine learning pipeline:

```text
Tech/
├── src/
│   ├── dataset.py     # Temporal split, scaler persistence/reuse, sliding windows & dataloaders
│   ├── app.py         # FastAPI dashboard for snapshot, CSV batch inference and RAG chat
│   ├── inference_service.py # Shared snapshot/batch inference logic and session helpers
│   ├── rag_service.py # Shared RAG runtime for CLI and FastAPI chat
│   ├── model.py       # MLP and GRU classifiers with defensive hyperparameter validation
│   ├── train.py       # MLP training loop with class-weighted loss and Early Stopping on F1
│   ├── train_gru.py   # GRU training entry point sharing the same training/eval machinery
│   ├── templates/     # Jinja2 frontend for the local dashboard
│   ├── training_common.py # Shared loop utilities for MLP/GRU training
│   ├── evaluate.py    # Checkpoint loading, persisted-scaler inference & confusion matrix
│   └── sweep.py       # Exhaustive hyperparameter grid search with threshold tuning
├── tests/
│   ├── test_model.py  # Unit + integration tests for model, data pipeline and CLI artefacts
│   └── test_app.py    # FastAPI health/predict tests
├── generate_data.py   # Physics-based synthetic data generator
└── README.md          # Technical documentation
```

---

## 🌡️ 1. Physics-Based Data Simulation (`generate_data.py`)

To evaluate the system, we simulate **10,000 sequential records** of server sensors (at 1-minute intervals). Rather than using uncorrelated random noise, the generator enforces physical laws to create realistic time-series dependencies:

1. **CPU & Memory Usage:** Modelled as autoregressive processes (AR-1) with stochastic load spikes to simulate realistic user traffic bursts.
2. **Network Traffic:** Positively correlated with CPU usage ($0.6 \times \text{CPU}$) plus exponential noise.
3. **CPU Temperature:** Dynamics are modeled using a discrete approximation of **Newton's Law of Cooling**:
   $$T_t = T_{t-1} + \alpha \cdot \text{cpu\_load}_t - \beta \cdot (T_{t-1} - T_{\text{ambient}}) + \epsilon_t$$
   Where $T_{\text{ambient}} = 35^\circ\text{C}$, $\alpha$ is the heating rate of the CPU, and $\beta$ is the cooling rate of the heatsink. This simulates **thermal inertia** (heat builds up gradually).
4. **Causal Failure Logic:** A failure (`failure = 1`) occurs with a 90% probability if the CPU temperature exceeds $80^\circ\text{C}$ and memory usage is $> 85\%$ for **3 or more consecutive steps**. Random failure noise is kept at $0.1\%$ to ensure a high signal-to-noise ratio.
5. **Class Imbalance:** Out of 10,000 records, exactly **508 are anomalies (5.08%)**, reflecting a realistic anomaly rate in production systems.

---

## 📋 2. Data Pipeline & Sliding Window (`src/dataset.py`)

* **Temporal Split:** To prevent *look-ahead bias* (data leakage), the sequential dataset is split chronologically into **70% Train (7,000 rows)**, **15% Validation (1,500 rows)**, and **15% Test (1,500 rows)**. No random shuffling is performed prior to splitting.
* **Feature Scaling:** A `MinMaxScaler` is fitted **only on the training split** and persisted as `artifacts/scaler.joblib`. Evaluation and inference reload that persisted scaler instead of silently refitting a new one, preserving consistency between training and offline validation.
* **Sliding Window:** Because system failures depend on a cumulative 3-step thermal overload, the pipeline groups the last 5 time-steps $[t-4, t-3, t-2, t-1, t]$ to predict the state at time $t$. The final input tensor for each sample is flattened into **20 dimensions** ($5 \text{ steps} \times 4 \text{ features}$).

---

## 🧠 3. PyTorch Model Architecture (`src/model.py`)

The deep learning model is an optimized Multi-Layer Perceptron (MLP) implemented in PyTorch (`AnomalyDetectorMLP`):
* **Input Layer:** 20 dimensions (flattened sliding window).
* **Hidden Layer 1:** 64 neurons $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)`.
* **Hidden Layer 2:** 32 neurons $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)`.
* **Output Layer:** 1 neuron returning raw **logits** (no sigmoid applied at the output).
* **Loss Function:** `BCEWithLogitsLoss` is used for numerical stability. Class imbalance is handled by setting `pos_weight` dynamically from the train split counts (raw pos_weight or "pos_weight crudo" $\approx 18.82$, scaled by $0.25$ to an effective pos_weight or "pos_weight efectivo" of $4.70$ after the grid search).
* **Defensive Validation:** Invalid dimensions or dropout values are rejected at construction time to fail fast on bad experiment configurations.

---

## 🏋️ 4. Training, Sweep & Threshold Calibration (`src/train.py` & `src/sweep.py`)

### Hyperparameter Sweep Results:
An exhaustive grid search of **162 hyperparameter combinations** was executed on both an Apple Silicon M4 GPU (using `mps`) and a workstation with an NVIDIA RTX 4070 Super (using `cuda`). 

#### 🚨 Crucial System Finding: Compute vs. I/O-Bound Bottleneck
The sweep execution times were:
* **Mac M4 (MPS):** **16.1 minutes**
* **PC RTX 4070 Super (CUDA):** **17.8 minutes**

*Analysis:* Since the MLP model is tiny (3,649 parameters) and the dataset easily fits in memory, the compute requirements are negligible. The time is dominated by memory transfers between CPU and GPU (`to(device)`) and Python interpreter overhead. This shows that the training is **I/O-bound**, and parallel GPU computing does not speed it up compared to Apple's unified memory architecture.

#### ⚠️ MPS Pitfall: `non_blocking=True` Corrupts Tensors
While integrating PyTorch with Apple Silicon (`mps`), we observed that calling `.to(device, non_blocking=True)` inside the training loop produced corrupted loss values (`loss.item()` returned magnitudes of ~1e23). The optimization is a documented CUDA convention that **does not behave the same way on MPS**: with overlapped CPU→MPS transfers, the kernel reads partially-written buffers. The fix is to drop the flag — plain `tensor.to(device)` is correct and only marginally slower because of unified memory. This is recorded in the project's development notes so future developers on the project avoid the same trap.

#### Grid Search Optimization:
The sweep optimized the model by scanning the precision-recall curve on the validation set to find the **best decision threshold** (instead of a static 0.5):

| Metric | Baseline Configuration (thr=0.5) | Optimized Configuration (thr=0.71) |
| :--- | :---: | :---: |
| **Validation F1-Score** | 0.647 | **0.854** |
| **Validation Precision** | 0.48 | **0.850** |
| **Validation Recall** | 0.98 | **0.850** |
| **Optimal Threshold** | 0.50 | **0.710** |
| **Effective pos_weight** | 18.82 | **4.70** (Scale 0.25) |
| **Hidden Layers** | (64, 32) | **(64, 32)** *(unchanged — sweep confirmed the original architecture)* |
| **Learning rate** | 1e-3 | **3e-3** |
| **Dropout** | 0.30 | **0.20** |
| **Batch size** | 128 | **64** |

---

## 🧪 5. Evaluation & Test Robustness (`src/evaluate.py` & `tests/test_model.py`)

* **Test Generalization:** When evaluated on the unseen Test Split (the final 1,500 minutes of the server's timeline) using the optimal threshold of **0.71**, the model achieved:
  * **Test F1-Score:** **0.8384**
  * **Test Precision:** **0.8469**
  * **Test Recall:** **0.8300**
  
  The marginal difference between validation ($0.854$) and test ($0.838$) metrics confirms that the model generalizes robustly and does not suffer from overfitting.
* **Confusion Matrix:** The resulting confusion matrix (`test_confusion_matrix.png`) shows an outstanding balance, capturing almost all thermal failures while keeping false alarms to a minimum.
* **Testing:** A suite of 21 tests is implemented in `tests/test_model.py` and run via `pytest`. It now covers model I/O shapes, invalid hyperparameter rejection, causal sliding-window construction, sequential temporal splitting, scaler fit isolation, persisted-scaler reuse during inference, and a short end-to-end CLI run that trains, saves, reloads and evaluates a checkpoint.

---

## 🚀 How to Install and Run the Project

### 1. Set Up Environment
Ensure you have Python 3.12+ installed. Create and activate a virtual environment, then install the dependencies:
```bash
# Create environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate  # On macOS/Linux
# or: .\.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

Optional shortcuts for the full demo are available in [Makefile](Makefile).

### 2. Generate the Dataset
Create the synthetic telemetry data with the physical sensors logic:
```bash
python generate_data.py
```

### 3. Run the Hyperparameter Sweep
Perform the grid search to tune the parameters and decision thresholds:
```bash
python -m src.sweep --epochs 30 --patience 7
```

### 4. Train the Final Model
Train the final model with the optimal parameters found by the sweep:
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

### 5. Evaluate on Test Split
Generate final metrics and the confusion matrix (the threshold is read from the checkpoint by default):
```bash
python -m src.evaluate --checkpoint best_model.pth
```

### 6. Run the Local FastAPI Demo
Start the dashboard once `best_model.pth` and `artifacts/scaler.joblib` exist:
```bash
uvicorn src.app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The dashboard now includes:

* **Snapshot inference** with 4 sliders. The UI repeats the same reading 5 times to satisfy the model's temporal window. It is intentionally labelled as approximate.
* **Batch CSV upload** for files containing `timestamp`, `cpu_usage`, `mem_usage`, `network_traffic`, `cpu_temp`.
* **Temporal chart** with CPU temperature, CPU usage and red anomaly markers using Chart.js.
* **CSV download** of analyzed results, including `probability` and `prediction`.
* **Integrated documentation chat** backed by the local FAISS index and Ollama-based RAG pipeline.

Batch mode reuses the trained scaler. If uploaded values fall outside the scaler's learned range, inputs are clipped to `[0, 1]` and the UI shows a warning.

The dashboard will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Docker Delivery

If you want evaluators to run the app in a more reproducible way, you can also ship it with Docker:

```bash
docker compose up --build
```

or:

```bash
make docker-run
```

Important: the web app still needs the trained artefacts to exist inside the delivery bundle:

* `best_model.pth`
* `artifacts/scaler.joblib`

If you do **not** include those artefacts, the reviewer must generate them first with training commands before opening the web demo.

### 7. Run Tests
Run the pytest suite to verify model dimensions, data pipeline and FastAPI endpoints:
```bash
uv run pytest tests/
```

## Demo Shortcuts

```bash
make test
make evaluate
make app
make rag
make docker-build
make docker-run
```

---

## 🧪 6. GRU Variant & Experimental Comparison (`src/train_gru.py`)

As an extension to the baseline MLP, I trained a recurrent **GRU-based classifier** (`AnomalyDetectorGRU`) to test whether explicitly modeling the 5-step temporal sequence improved anomaly detection.

### Architectural and Metric Comparison

The comparison below reflects the metrics of the checkpoints currently shipped with the repository:

| Metric / Feature | MLP Baseline | GRU Variant |
| :--- | :---: | :---: |
| **Parameters** | 3,649 | 13,505 |
| **Structure** | Flat sliding window (20 dimensions) | Sequential recurrent `[batch, 5, 4]` |
| **Decision Threshold** | 0.710 | 0.613 |
| **Best Validation F1** | **0.8545** | **0.8393** |
| **Test F1-Score** | **0.8384** | **0.8155** |
| **Test Precision** | **0.8469** | **0.7925** |
| **Test Recall** | **0.8300** | **0.8400** |

### Experimental Findings & Conclusions
* **Short Sequence Limit:** With only 5 minutes of context, the recurrent inductive bias of the GRU does not translate into a measurable advantage over the simpler flattened window approach.
* **Precision vs. Recall Trade-off:** The GRU slightly increases recall (`0.8400` vs `0.8300`), but loses enough precision (`0.7925` vs `0.8469`) to produce a lower overall F1-score on test.
* **Final Verdict:** The MLP remains the strongest default model for this challenge because it is smaller, simpler, and more stable on the held-out split.
* **Code Separation:** Both training pathways remain available (`src/train.py` for MLP and `src/train_gru.py` for GRU) while sharing the common training utilities in `src/training_common.py`.
