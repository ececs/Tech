"""Reusable RAG service shared by the CLI and FastAPI app."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_DOC_PATHS: tuple[Path, ...] = (
    PROJECT_ROOT / "README.md",
)
DEFAULT_INDEX_DIR: Path = PROJECT_ROOT / "artifacts" / "rag_index"
DEFAULT_OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://192.168.31.181:11434")
DEFAULT_OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
DEFAULT_LOCAL_OLLAMA_URL: str = os.getenv("OLLAMA_LOCAL_URL", "http://127.0.0.1:11434")
DEFAULT_FALLBACK_MODEL: str = os.getenv("OLLAMA_FALLBACK_MODEL", "tinyllama")
# Multilingual embedding model: the project corpus mixes Spanish
# (DIARIO_PROYECTO.md) with English (README.md). The English-only
# all-MiniLM-L6-v2 was tried first and FAILED to retrieve key bilingual
# passages — switching to the multilingual paraphrase model fixed it.
# Keep it as the project-wide default.
DEFAULT_EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
DEFAULT_TOP_K = 6

# Deterministic intent router. Trivial conversational primitives are
# answered before invoking the LLM: it removes a ~12 s remote call and
# eliminates any chance of the model going off-script on a greeting.
# The full RAG pipeline (with the strict grounding prompt) handles
# everything else unchanged.
GREETING_PATTERNS: frozenset[str] = frozenset({
    "hola",
    "buenas",
    "buenos dias",
    "buenos días",
    "buenas tardes",
    "buenas noches",
    "hey",
    "hi",
    "hello",
    "saludos",
    "que tal",
    "qué tal",
    "que hay",
    "qué hay",
})
THANKS_PATTERNS: frozenset[str] = frozenset({
    "gracias",
    "muchas gracias",
    "mil gracias",
    "thanks",
    "thank you",
    "thx",
})
HELP_PATTERNS: frozenset[str] = frozenset({
    "ayuda",
    "help",
    "que puedes hacer",
    "qué puedes hacer",
    "que sabes",
    "qué sabes",
    "como funcionas",
    "cómo funcionas",
})

GREETING_RESPONSE = (
    "¡Hola! Soy el asistente técnico de este proyecto. "
    "Pregúntame sobre la arquitectura, métricas, hiperparámetros "
    "o cualquier detalle de la documentación."
)
THANKS_RESPONSE = "De nada. ¿Algo más sobre el proyecto?"
HELP_RESPONSE = (
    "Puedo responder preguntas sobre el modelo MLP/GRU, el pipeline de "
    "datos, los resultados en test, el sweep de hiperparámetros y la "
    "arquitectura del sistema. Cito siempre el archivo fuente."
)


def maybe_route_intent(question: str) -> str | None:
    """Return a fixed reply for conversational primitives, or ``None``.

    Detection is exact-match on a normalized form of the question
    (lowercased and trimmed of leading/trailing punctuation/whitespace).
    Anything that doesn't match a known intent falls through to the
    full RAG pipeline.

    Args:
        question: The user's raw input string.

    Returns:
        A canned response string if the input matches a known intent,
        otherwise ``None``.
    """
    normalized = question.strip().lower().strip("?¿!¡.,;: \t\n")
    if not normalized:
        return None
    if normalized in GREETING_PATTERNS:
        return GREETING_RESPONSE
    if normalized in THANKS_PATTERNS:
        return THANKS_RESPONSE
    if normalized in HELP_PATTERNS:
        return HELP_RESPONSE
    return None


PROMPT_TEMPLATE = """Eres un asistente técnico que responde preguntas sobre un proyecto de Deep Learning para detección de anomalías en servidores. Usa EXCLUSIVAMENTE la información de los pasajes a continuación para responder. Si la respuesta no está en los pasajes, di "No tengo esa información en la documentación". Responde en español, de forma concisa (máximo 6 líneas) y cita el archivo fuente entre paréntesis al final de cada afirmación, por ejemplo "(README.md)".

Pasajes recuperados:
---
{context}
---

Pregunta: {question}

Respuesta:"""


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved document chunk with its source metadata."""

    source: str
    start_line: int
    snippet: str
    score: float


@dataclass(frozen=True)
class RAGResources:
    """RAG runtime objects loaded once and reused by callers."""

    vectorstore: FAISS | None
    llm: OllamaLLM | None
    ollama_url: str
    fallback_model: str | None
    top_k: int
    ready: bool
    error: str | None = None


def load_documents(paths: tuple[Path, ...]) -> list[Document]:
    """Read each Markdown file and wrap it in a LangChain ``Document``."""
    docs: list[Document] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "path": str(path)},
            )
        )
    return docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks: list[Document] = []
    for document in documents:
        text = document.page_content
        search_start = 0
        for chunk_text in splitter.split_text(text):
            offset = text.find(chunk_text, search_start)
            if offset == -1:
                offset = search_start
            start_line = text.count("\n", 0, offset) + 1
            metadata = dict(document.metadata)
            metadata["start_line"] = start_line
            chunks.append(Document(page_content=chunk_text, metadata=metadata))
            search_start = offset + max(len(chunk_text) - chunk_overlap, 1)
    return chunks


def build_or_load_index(
    doc_paths: tuple[Path, ...],
    index_dir: Path,
    embedding_model: str,
    rebuild: bool = False,
) -> FAISS:
    """Build the FAISS index from scratch or load the cached one."""
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if index_dir.exists() and not rebuild:
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    documents = load_documents(doc_paths)
    chunks = chunk_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def retrieve(
    vectorstore: FAISS, question: str, k: int = DEFAULT_TOP_K
) -> list[RetrievedChunk]:
    """Return the top-k chunks most similar to ``question``."""
    results = vectorstore.similarity_search_with_score(question, k=k)
    return [
        RetrievedChunk(
            source=str(doc.metadata.get("source", "unknown")),
            start_line=int(doc.metadata.get("start_line", 1)),
            snippet=doc.page_content.strip(),
            score=float(score),
        )
        for doc, score in results
    ]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered context block."""
    return "\n\n".join(
        f"[{i}] ({chunk.source}:{chunk.start_line})\n{chunk.snippet}"
        for i, chunk in enumerate(chunks, start=1)
    )


def answer(
    llm: OllamaLLM,
    vectorstore: FAISS,
    question: str,
    k: int = DEFAULT_TOP_K,
) -> tuple[str, list[RetrievedChunk]]:
    """Retrieve context and generate an answer with the configured LLM."""
    chunks = retrieve(vectorstore, question, k=k)
    prompt = PROMPT_TEMPLATE.format(
        context=format_context(chunks),
        question=question,
    )
    response = llm.invoke(prompt)
    return response.strip(), chunks


def is_remote_ollama_unreachable(exc: Exception, ollama_url: str) -> bool:
    """Heuristically detect connectivity failures to the configured Ollama URL."""
    message = str(exc).lower()
    host = urlparse(ollama_url).hostname or ""
    connectivity_markers = (
        "connection refused",
        "connecterror",
        "connection error",
        "failed to connect",
        "all connection attempts failed",
        "name or service not known",
        "nodename nor servname provided",
    )
    return host in message or any(marker in message for marker in connectivity_markers)


def answer_with_fallback(
    llm: OllamaLLM,
    vectorstore: FAISS,
    question: str,
    k: int,
    ollama_url: str,
    fallback_model: str | None,
) -> tuple[str, list[RetrievedChunk]]:
    """Answer with the primary remote Ollama model and fallback if unreachable."""
    try:
        return answer(llm, vectorstore, question, k=k)
    except Exception as exc:  # noqa: BLE001
        if not fallback_model or not is_remote_ollama_unreachable(exc, ollama_url):
            raise
        fallback_llm = OllamaLLM(
            base_url=DEFAULT_LOCAL_OLLAMA_URL,
            model=fallback_model,
            temperature=0.2,
        )
        try:
            return answer(fallback_llm, vectorstore, question, k=k)
        except Exception as fallback_exc:  # noqa: BLE001
            raise RuntimeError(
                "Ollama remoto inaccesible y el fallback local no está disponible. "
                "Verifica el servidor remoto o arranca un modelo local como tinyllama."
            ) from fallback_exc


def load_rag_resources(
    doc_paths: tuple[Path, ...] = DEFAULT_DOC_PATHS,
    index_dir: Path = DEFAULT_INDEX_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
    top_k: int = DEFAULT_TOP_K,
    rebuild: bool = False,
) -> RAGResources:
    """Load vector store and LLM client once for reuse across requests."""
    try:
        vectorstore = build_or_load_index(
            doc_paths=doc_paths,
            index_dir=index_dir,
            embedding_model=embedding_model,
            rebuild=rebuild,
        )
        llm = OllamaLLM(
            base_url=ollama_url,
            model=ollama_model,
            temperature=0.2,
        )
        return RAGResources(
            vectorstore=vectorstore,
            llm=llm,
            ollama_url=ollama_url,
            fallback_model=fallback_model,
            top_k=top_k,
            ready=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load RAG resources: %s", exc)
        return RAGResources(
            vectorstore=None,
            llm=None,
            ollama_url=ollama_url,
            fallback_model=fallback_model,
            top_k=top_k,
            ready=False,
            error=str(exc),
        )


def ask_question(
    question: str,
    resources: RAGResources,
) -> tuple[str, list[RetrievedChunk]]:
    """Run a RAG question against preloaded resources.

    Conversational primitives (greetings, thanks, capability queries)
    are short-circuited by :func:`maybe_route_intent` before any
    retrieval or LLM call, so the system replies instantly and cannot
    hallucinate on them. Everything else falls through to the strict
    RAG flow.
    """
    routed = maybe_route_intent(question)
    if routed is not None:
        return routed, []

    if not resources.ready or resources.vectorstore is None or resources.llm is None:
        raise RuntimeError(
            "El servicio RAG no está listo. "
            f"Detalle: {resources.error or 'recursos no cargados'}"
        )
    return answer_with_fallback(
        llm=resources.llm,
        vectorstore=resources.vectorstore,
        question=question,
        k=resources.top_k,
        ollama_url=resources.ollama_url,
        fallback_model=resources.fallback_model,
    )
