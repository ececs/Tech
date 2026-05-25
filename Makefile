PYTHON ?= python3
UV ?= uv

.PHONY: install test train train-gru evaluate app rag docker-build docker-run

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(UV) run pytest tests/

train:
	$(PYTHON) -m src.train --epochs 50 --patience 7 --batch-size 64 --hidden-dims 64 32 --lr 3e-3 --dropout 0.2 --pos-weight-scale 0.25 --decision-threshold 0.71

train-gru:
	$(PYTHON) -m src.train_gru --epochs 60 --patience 12 --batch-size 64 --lr 1e-3 --dropout 0.2 --hidden-size 64 --num-layers 1 --pos-weight-scale 0.25 --decision-threshold 0.613

evaluate:
	$(PYTHON) -m src.evaluate --checkpoint best_model.pth

app:
	uvicorn src.app:app --reload

rag:
	$(PYTHON) -m src.rag_assistant --interactive

docker-build:
	docker build -t techstream-demo .

docker-run:
	docker compose up --build
