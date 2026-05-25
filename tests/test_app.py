from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from src.app import RuntimeResources, create_app
from src.rag_service import RAGResources


class DummySnapshotModel(nn.Module):
    """Small deterministic model for FastAPI endpoint tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_value = x.mean(dim=1, keepdim=True)
        return (mean_value - 0.5) * 4.0


class DummyRagLLM:
    def invoke(self, _prompt: str) -> str:
        return "El threshold final calibrado es 0.71 (README.md)."


@pytest.fixture
def ready_runtime() -> RuntimeResources:
    scaler = MinMaxScaler()
    scaler.fit(
        np.array(
            [
                [0.0, 0.0, 0.0, 20.0],
                [100.0, 100.0, 250.0, 120.0],
            ],
            dtype=np.float64,
        )
    )
    model = DummySnapshotModel().eval()
    return RuntimeResources(
        model=model,
        scaler=scaler,
        threshold=0.5,
        device=torch.device("cpu"),
        model_type="mlp",
        ready=True,
    )


@pytest.fixture
def ready_rag() -> RAGResources:
    return RAGResources(
        vectorstore=object(),
        llm=DummyRagLLM(),
        ollama_url="http://dummy",
        fallback_model=None,
        top_k=2,
        ready=True,
    )


def test_health_ok(ready_runtime: RuntimeResources, ready_rag: RAGResources) -> None:
    client = TestClient(create_app(runtime_override=ready_runtime, rag_override=ready_rag))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "mlp"
    assert response.json()["device"] == "cpu"


def test_predict_snapshot(ready_runtime: RuntimeResources, ready_rag: RAGResources) -> None:
    client = TestClient(create_app(runtime_override=ready_runtime, rag_override=ready_rag))

    response = client.post(
        "/predict",
        json={
            "cpu_usage": 95.0,
            "mem_usage": 90.0,
            "network_traffic": 220.0,
            "cpu_temp": 102.0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["prediction"] in {0, 1}
    assert payload["decision"]


def test_predict_window_uses_real_temporal_sequence(
    ready_runtime: RuntimeResources, ready_rag: RAGResources
) -> None:
    """The /predict_window endpoint must accept up to 5 chronological readings."""
    client = TestClient(create_app(runtime_override=ready_runtime, rag_override=ready_rag))

    response = client.post(
        "/predict_window",
        json={
            "readings": [
                {"cpu_usage": 30, "mem_usage": 40, "network_traffic": 50, "cpu_temp": 60},
                {"cpu_usage": 50, "mem_usage": 55, "network_traffic": 70, "cpu_temp": 75},
                {"cpu_usage": 80, "mem_usage": 88, "network_traffic": 110, "cpu_temp": 92},
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "trend"
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["prediction"] in {0, 1}


def test_predict_window_rejects_empty_payload(
    ready_runtime: RuntimeResources, ready_rag: RAGResources
) -> None:
    client = TestClient(create_app(runtime_override=ready_runtime, rag_override=ready_rag))
    response = client.post("/predict_window", json={"readings": []})
    assert response.status_code == 422  # pydantic min_length violation


def test_predict_window_rejects_too_many_readings(
    ready_runtime: RuntimeResources, ready_rag: RAGResources
) -> None:
    client = TestClient(create_app(runtime_override=ready_runtime, rag_override=ready_rag))
    reading = {"cpu_usage": 50, "mem_usage": 50, "network_traffic": 50, "cpu_temp": 50}
    response = client.post(
        "/predict_window", json={"readings": [reading] * 6}
    )
    assert response.status_code == 422  # pydantic max_length violation


def test_simulation_scenarios_lists_curated_windows(
    ready_runtime: RuntimeResources, ready_rag: RAGResources
) -> None:
    client = TestClient(create_app(runtime_override=ready_runtime, rag_override=ready_rag))
    response = client.get("/simulation/scenarios")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) >= 3
    # The recommended ramp-up scenario must always be present.
    ids = {scenario["id"] for scenario in payload}
    assert "ramp_up" in ids
    for scenario in payload:
        assert {"id", "label", "description", "start", "length"} <= scenario.keys()
        assert scenario["length"] > 0


def test_simulation_stream_returns_rows_with_ground_truth(
    ready_runtime: RuntimeResources, ready_rag: RAGResources
) -> None:
    client = TestClient(create_app(runtime_override=ready_runtime, rag_override=ready_rag))
    response = client.get("/simulation/stream", params={"start": 8600, "length": 20})
    assert response.status_code == 200
    payload = response.json()
    assert payload["start"] == 8600
    assert payload["length"] == 20
    assert len(payload["rows"]) == 20
    sample = payload["rows"][0]
    assert {
        "timestamp",
        "cpu_usage",
        "mem_usage",
        "network_traffic",
        "cpu_temp",
        "failure_actual",
    } <= sample.keys()
    assert sample["failure_actual"] in {0, 1}


def test_analyze_preset_runs_scenario_through_batch_pipeline(
    ready_runtime: RuntimeResources, ready_rag: RAGResources, tmp_path
) -> None:
    client = TestClient(
        create_app(
            runtime_override=ready_runtime,
            rag_override=ready_rag,
            sessions_dir_override=tmp_path,
        )
    )
    response = client.post("/analyze_preset", json={"scenario_id": "ramp_up"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"]
    assert payload["summary"]["n_rows"] > 0
    assert len(payload["records"]) == payload["summary"]["n_rows"]
    # The analyzed CSV must be downloadable through the regular endpoint.
    download = client.post(
        "/download_results", json={"session_id": payload["session_id"]}
    )
    assert download.status_code == 200
    assert "probability" in download.text


def test_analyze_preset_rejects_unknown_scenario(
    ready_runtime: RuntimeResources, ready_rag: RAGResources
) -> None:
    client = TestClient(create_app(runtime_override=ready_runtime, rag_override=ready_rag))
    response = client.post("/analyze_preset", json={"scenario_id": "does_not_exist"})
    assert response.status_code == 404


def test_simulation_stream_rejects_invalid_bounds(
    ready_runtime: RuntimeResources, ready_rag: RAGResources
) -> None:
    client = TestClient(create_app(runtime_override=ready_runtime, rag_override=ready_rag))

    for params in [
        {"start": -1, "length": 50},
        {"start": 8600, "length": 0},
        {"start": 8600, "length": 9000},  # > 2000 cap
        {"start": 999_999, "length": 10},
    ]:
        response = client.get("/simulation/stream", params=params)
        assert response.status_code == 400, params


def test_predict_returns_503_when_runtime_not_ready(ready_rag: RAGResources) -> None:
    degraded_runtime = RuntimeResources(
        model=None,
        scaler=None,
        threshold=0.5,
        device=torch.device("cpu"),
        model_type="mlp",
        ready=False,
        error="missing artifacts",
    )
    client = TestClient(create_app(runtime_override=degraded_runtime, rag_override=ready_rag))

    response = client.post(
        "/predict",
        json={
            "cpu_usage": 50.0,
            "mem_usage": 50.0,
            "network_traffic": 50.0,
            "cpu_temp": 50.0,
        },
    )

    assert response.status_code == 503
    assert "missing artifacts" in response.json()["detail"]


def test_upload_csv_and_download_results(
    ready_runtime: RuntimeResources,
    ready_rag: RAGResources,
    tmp_path,
) -> None:
    client = TestClient(
        create_app(
            runtime_override=ready_runtime,
            rag_override=ready_rag,
            sessions_dir_override=tmp_path,
            session_limit_override=5,
        )
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="min"),
            "cpu_usage": [20, 35, 40, 50, 60, 85],
            "mem_usage": [25, 30, 45, 52, 58, 88],
            "network_traffic": [15, 30, 55, 70, 80, 220],
            "cpu_temp": [40, 45, 55, 65, 78, 102],
        }
    )
    csv_bytes = frame.to_csv(index=False).encode("utf-8")

    upload_response = client.post(
        "/upload_csv",
        files={"file": ("telemetry.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["session_id"]
    assert payload["summary"]["n_rows"] == 6
    assert len(payload["records"]) == 6

    download_response = client.post(
        "/download_results",
        json={"session_id": payload["session_id"]},
    )

    assert download_response.status_code == 200
    assert "probability" in download_response.text
    assert "prediction" in download_response.text


def test_ask_endpoint(ready_runtime: RuntimeResources, ready_rag: RAGResources, monkeypatch) -> None:
    client = TestClient(
        create_app(
            runtime_override=ready_runtime,
            rag_override=ready_rag,
        )
    )

    from src import app as app_module
    from src.rag_service import RetrievedChunk

    def fake_ask_question(question: str, _resources: RAGResources):
        assert "threshold" in question.lower()
        return (
            "El threshold final calibrado es 0.71 (README.md).",
            [
                RetrievedChunk(
                    source="README.md",
                    start_line=1,
                    snippet="Optimal Threshold | 0.710",
                    score=0.12,
                )
            ],
        )

    monkeypatch.setattr(app_module, "ask_question", fake_ask_question)

    response = client.post("/ask", json={"question": "¿Qué threshold final se usa?"})

    assert response.status_code == 200
    payload = response.json()
    assert "0.71" in payload["answer"]
    assert payload["citations"][0]["source"] == "README.md"


def test_intent_router_returns_canned_responses() -> None:
    """The intent router resolves greetings/thanks/help without invoking the LLM."""
    from src.rag_service import (
        GREETING_RESPONSE,
        HELP_RESPONSE,
        THANKS_RESPONSE,
        maybe_route_intent,
    )

    greeting_inputs = ["hola", "Hola!", "¡Hola!", "  HOLA  ", "buenas", "Buenos días"]
    for prompt in greeting_inputs:
        assert maybe_route_intent(prompt) == GREETING_RESPONSE, prompt

    for prompt in ["gracias", "Muchas gracias", "thanks!"]:
        assert maybe_route_intent(prompt) == THANKS_RESPONSE, prompt

    for prompt in ["ayuda", "¿qué puedes hacer?", "Cómo funcionas"]:
        assert maybe_route_intent(prompt) == HELP_RESPONSE, prompt

    # Non-trivial questions must fall through to the full RAG flow.
    for prompt in [
        "¿Qué F1 alcanza el modelo?",
        "hola, dime el F1",
        "hola dime el F1",
        "",
    ]:
        assert maybe_route_intent(prompt) is None, prompt


def test_ask_short_circuits_on_greeting(
    ready_runtime: RuntimeResources, ready_rag: RAGResources
) -> None:
    """A bare greeting must return the canned reply with empty citations and never hit the LLM."""
    from src.rag_service import GREETING_RESPONSE

    client = TestClient(
        create_app(runtime_override=ready_runtime, rag_override=ready_rag)
    )

    response = client.post("/ask", json={"question": "hola"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == GREETING_RESPONSE
    assert payload["citations"] == []


def test_get_image_success(ready_runtime: RuntimeResources, ready_rag: RAGResources) -> None:
    client = TestClient(create_app(runtime_override=ready_runtime, rag_override=ready_rag))
    response = client.get("/images/distribution_clases.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_get_image_not_allowed(ready_runtime: RuntimeResources, ready_rag: RAGResources) -> None:
    client = TestClient(create_app(runtime_override=ready_runtime, rag_override=ready_rag))
    
    # Non-existent image
    response = client.get("/images/does_not_exist.png")
    assert response.status_code == 404
    
    # Existing file but not allowed (e.g. generate_data.py)
    response = client.get("/images/generate_data.py")
    assert response.status_code == 404
