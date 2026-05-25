"""FastAPI dashboard for snapshot and CSV batch anomaly inference."""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from src.inference_service import (
    BatchInferenceResult,
    RuntimeResources,
    compute_session_id,
    infer_uploaded_dataframe,
    load_runtime_resources,
    predict_snapshot,
)
from src.rag_service import RAGResources, ask_question, load_rag_resources

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
TEMPLATES_DIR: Path = Path(__file__).resolve().parent / "templates"
SESSIONS_DIR: Path = PROJECT_ROOT / "artifacts" / "sessions"
SESSION_LIMIT = 5


class PredictRequest(BaseModel):
    """Incoming telemetry snapshot used to build a synthetic 5-step window."""

    cpu_usage: float = Field(..., ge=0.0, le=100.0)
    mem_usage: float = Field(..., ge=0.0, le=100.0)
    network_traffic: float = Field(..., ge=0.0)
    cpu_temp: float = Field(..., ge=0.0, le=150.0)


class PredictResponse(BaseModel):
    """Prediction payload returned by the FastAPI endpoint."""

    model_config = ConfigDict(from_attributes=True)

    probability: float
    prediction: int
    threshold: float
    decision: str


class HealthResponse(BaseModel):
    """Basic service health and model readiness."""

    status: str
    device: str
    model: str
    ready: bool
    error: str | None = None


class SummaryPeak(BaseModel):
    timestamp: str
    cpu_temp: float


class UploadSummary(BaseModel):
    n_rows: int
    n_anomalies: int
    anomaly_rate: float
    top_thermal_peaks: list[SummaryPeak]


class UploadRecord(BaseModel):
    idx: int
    timestamp: str
    cpu_usage: float
    cpu_temp: float
    prob: float
    prediction: int


class UploadResponse(BaseModel):
    session_id: str
    summary: UploadSummary
    records: list[UploadRecord]
    warnings: list[str] = Field(default_factory=list)


class DownloadRequest(BaseModel):
    session_id: str = Field(..., min_length=6, max_length=64)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class CitationResponse(BaseModel):
    source: str
    start_line: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


def persist_session_dataframe(
    session_id: str,
    dataframe: pd.DataFrame,
    sessions_dir: Path,
) -> None:
    """Persist a processed dataframe, preferring parquet and falling back to pickle."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = sessions_dir / f"{session_id}.parquet"
    pickle_path = sessions_dir / f"{session_id}.pkl"
    try:
        dataframe.to_parquet(parquet_path, index=False)
        if pickle_path.exists():
            pickle_path.unlink()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not persist session %s as parquet (%s). Falling back to pickle.",
            session_id,
            exc,
        )
        dataframe.to_pickle(pickle_path)
        if parquet_path.exists():
            parquet_path.unlink()


def load_session_dataframe(session_id: str, sessions_dir: Path) -> pd.DataFrame:
    """Load a processed dataframe persisted by :func:`persist_session_dataframe`."""
    parquet_path = sessions_dir / f"{session_id}.parquet"
    pickle_path = sessions_dir / f"{session_id}.pkl"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    raise FileNotFoundError(f"Session {session_id!r} not found")


def purge_old_sessions(sessions_dir: Path, keep: int = SESSION_LIMIT) -> None:
    """Keep only the newest processed sessions on disk."""
    if not sessions_dir.exists():
        return
    session_files = sorted(
        [
            path
            for path in sessions_dir.iterdir()
            if path.suffix in {".parquet", ".pkl"}
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale_path in session_files[keep:]:
        stale_path.unlink(missing_ok=True)


def build_predict_response(probability: float, threshold: float) -> PredictResponse:
    """Translate a probability score into the public API response shape."""
    prediction = int(probability > threshold)
    decision = (
        "Anomaly risk detected"
        if prediction == 1
        else "System appears stable"
    )
    return PredictResponse(
        probability=probability,
        prediction=prediction,
        threshold=threshold,
        decision=decision,
    )


def create_app(
    runtime_override: RuntimeResources | None = None,
    rag_override: RAGResources | None = None,
    sessions_dir_override: Path | None = None,
    session_limit_override: int | None = None,
) -> FastAPI:
    """Build the FastAPI application with optional test runtime injection."""
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Production path: load real resources lazily when the server starts.
        # When overrides are supplied (tests, custom embedding flows) they are
        # already attached in `create_app` below, so the lifespan only loads
        # what is still missing.
        if not hasattr(app.state, "runtime"):
            app.state.runtime = load_runtime_resources()
        if not hasattr(app.state, "rag"):
            app.state.rag = load_rag_resources()
        if not hasattr(app.state, "sessions_dir"):
            app.state.sessions_dir = SESSIONS_DIR
        if not hasattr(app.state, "session_limit"):
            app.state.session_limit = SESSION_LIMIT
        yield

    app = FastAPI(
        title="TechStream Dashboard",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Pre-populate `app.state` when explicit overrides are provided. This is
    # what tests rely on, because FastAPI's TestClient only triggers the
    # lifespan when used as a context manager (`with TestClient(app) as c:`).
    # Setting state at construction time keeps both `TestClient(app)` and the
    # context-manager form working identically.
    if runtime_override is not None:
        app.state.runtime = runtime_override
    if rag_override is not None:
        app.state.rag = rag_override
    if sessions_dir_override is not None:
        app.state.sessions_dir = sessions_dir_override
    if session_limit_override is not None:
        app.state.session_limit = session_limit_override

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        runtime: RuntimeResources = request.app.state.runtime
        context = {
            "request": request,
            "device": runtime.device.type,
            "model": runtime.model_type,
            "threshold": runtime.threshold,
            "ready": runtime.ready,
            "error": runtime.error,
            "rag_ready": request.app.state.rag.ready,
            "rag_error": request.app.state.rag.error,
        }
        return templates.TemplateResponse("index.html", context)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        runtime: RuntimeResources = request.app.state.runtime
        return HealthResponse(
            status="ok" if runtime.ready else "degraded",
            device=runtime.device.type,
            model=runtime.model_type,
            ready=runtime.ready,
            error=runtime.error,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        runtime: RuntimeResources = request.app.state.runtime
        try:
            probability = predict_snapshot(
                cpu_usage=payload.cpu_usage,
                mem_usage=payload.mem_usage,
                network_traffic=payload.network_traffic,
                cpu_temp=payload.cpu_temp,
                runtime=runtime,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return build_predict_response(probability, runtime.threshold)

    @app.post("/upload_csv", response_model=UploadResponse)
    async def upload_csv(
        request: Request,
        file: UploadFile = File(...),
    ) -> UploadResponse:
        runtime: RuntimeResources = request.app.state.runtime
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV uploads are supported")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty")

        try:
            dataframe = pd.read_csv(io.BytesIO(file_bytes))
            batch_result: BatchInferenceResult = infer_uploaded_dataframe(
                dataframe,
                runtime,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("CSV upload failed: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to process uploaded CSV") from exc

        session_id = compute_session_id(file_bytes)
        sessions_dir: Path = request.app.state.sessions_dir
        persist_session_dataframe(session_id, batch_result.processed_df, sessions_dir)
        purge_old_sessions(sessions_dir, keep=request.app.state.session_limit)

        return UploadResponse(
            session_id=session_id,
            summary=UploadSummary(**batch_result.summary),
            records=[UploadRecord(**record) for record in batch_result.records],
            warnings=batch_result.warnings,
        )

    @app.post("/download_results")
    def download_results(
        payload: DownloadRequest,
        request: Request,
    ) -> StreamingResponse:
        sessions_dir: Path = request.app.state.sessions_dir
        try:
            dataframe = load_session_dataframe(payload.session_id, sessions_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        csv_buffer = io.StringIO()
        dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        filename = f"techstream_analysis_{payload.session_id}.csv"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers=headers,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest, request: Request) -> AskResponse:
        rag: RAGResources = request.app.state.rag
        try:
            answer_text, citations = ask_question(payload.question, rag)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG ask failed: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to answer question") from exc

        return AskResponse(
            answer=answer_text,
            citations=[
                CitationResponse(
                    source=chunk.source,
                    start_line=chunk.start_line,
                    snippet=chunk.snippet,
                )
                for chunk in citations
            ],
        )

    return app


app = create_app()
