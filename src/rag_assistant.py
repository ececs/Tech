"""Local RAG assistant over the TechStream project documentation.

Indexes the project's Markdown documents into a local FAISS store using
``sentence-transformers/all-MiniLM-L6-v2`` embeddings, then answers
questions in Spanish by retrieving the top-k relevant chunks and asking
the remote Ollama instance (``llama3.2``) to produce a grounded response
with verbatim citations.

Run from the repository root::

    # one-shot question
    python -m src.rag_assistant "¿Por qué el GRU pierde frente al MLP?"

    # interactive REPL
    python -m src.rag_assistant --interactive

    # force rebuild of the FAISS index
    python -m src.rag_assistant --rebuild "..."

The index is persisted under ``artifacts/rag_index/`` and reused across
calls. The Ollama endpoint defaults to ``http://192.168.31.181:11434``
(the documented secondary workstation) and can be overridden with
``--ollama-url``.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_DOC_PATHS: tuple[Path, ...] = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "DIARIO_PROYECTO.md",
    PROJECT_ROOT / "GUION_VIDEO.md",
)
DEFAULT_INDEX_DIR: Path = PROJECT_ROOT / "artifacts" / "rag_index"
DEFAULT_OLLAMA_URL: str = "http://192.168.31.181:11434"
DEFAULT_OLLAMA_MODEL: str = "llama3.2"
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PROMPT_TEMPLATE = """Eres un asistente técnico que responde preguntas sobre un proyecto de Deep Learning para detección de anomalías en servidores. Usa EXCLUSIVAMENTE la información de los pasajes a continuación para responder. Si la respuesta no está en los pasajes, di "No tengo esa información en la documentación". Responde en español, de forma concisa (máximo 6 líneas) y cita el archivo fuente entre paréntesis al final de cada afirmación, por ejemplo "(README.md)".

Pasajes recuperados:
---
{context}
---

Pregunta: {question}

Respuesta:"""


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with its source metadata."""

    source: str
    snippet: str
    score: float


def load_documents(paths: tuple[Path, ...]) -> list[Document]:
    """Read each Markdown file and wrap it in a LangChain ``Document``.

    Args:
        paths: Markdown files to ingest.

    Returns:
        A list of ``Document`` instances with ``source`` metadata.

    Raises:
        FileNotFoundError: If any path does not exist.
    """
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
        logger.info("Loaded %s (%d chars)", path.name, len(text))
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
    chunks = splitter.split_documents(documents)
    logger.info(
        "Split %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def build_or_load_index(
    doc_paths: tuple[Path, ...],
    index_dir: Path,
    embedding_model: str,
    rebuild: bool = False,
) -> FAISS:
    """Build the FAISS index from scratch or load the cached one.

    Args:
        doc_paths: Source Markdown files.
        index_dir: Directory where the FAISS files live.
        embedding_model: HuggingFace sentence-transformers checkpoint.
        rebuild: If ``True``, always rebuild even if a cache exists.

    Returns:
        A loaded ``FAISS`` instance ready for similarity search.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if index_dir.exists() and not rebuild:
        logger.info("Loading FAISS index from %s", index_dir)
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("Building FAISS index from %d documents", len(doc_paths))
    documents = load_documents(doc_paths)
    chunks = chunk_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    logger.info("Persisted FAISS index to %s", index_dir)
    return vectorstore


def retrieve(
    vectorstore: FAISS, question: str, k: int = 4
) -> list[RetrievedChunk]:
    """Return the top-k chunks most similar to ``question``.

    Args:
        vectorstore: FAISS index ready for search.
        question: User question.
        k: Number of chunks to retrieve.

    Returns:
        List of ``RetrievedChunk`` ordered by descending similarity.
    """
    results = vectorstore.similarity_search_with_score(question, k=k)
    chunks = [
        RetrievedChunk(
            source=str(doc.metadata.get("source", "unknown")),
            snippet=doc.page_content.strip(),
            score=float(score),
        )
        for doc, score in results
    ]
    return chunks


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered context block."""
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] ({chunk.source})\n{chunk.snippet}")
    return "\n\n".join(blocks)


def answer(
    llm: OllamaLLM,
    vectorstore: FAISS,
    question: str,
    k: int = 4,
) -> tuple[str, list[RetrievedChunk]]:
    """Retrieve context and generate an answer with the remote LLM.

    Args:
        llm: A ready ``OllamaLLM`` client.
        vectorstore: Loaded FAISS index.
        question: User question.
        k: Number of context chunks to feed the LLM.

    Returns:
        Tuple ``(answer_text, retrieved_chunks)``.
    """
    chunks = retrieve(vectorstore, question, k=k)
    context = format_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    logger.info("Invoking %s with %d context chunks", llm.model, len(chunks))
    response = llm.invoke(prompt)
    return response.strip(), chunks


def print_answer(
    question: str, response: str, chunks: list[RetrievedChunk]
) -> None:
    """Pretty-print the LLM answer plus retrieved sources."""
    print()
    print("Pregunta:")
    print(textwrap.indent(question, "  "))
    print()
    print("Respuesta:")
    print(textwrap.indent(response, "  "))
    print()
    print("Fuentes citadas (top-k chunks recuperados):")
    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.snippet.replace("\n", " ")
        if len(preview) > 140:
            preview = preview[:137] + "..."
        print(f"  [{i}] {chunk.source}  (score={chunk.score:.3f})")
        print(f"      {preview}")
    print()


def interactive_loop(
    llm: OllamaLLM, vectorstore: FAISS, k: int
) -> None:
    """Run a simple REPL until the user types ``exit`` or ``quit``."""
    print("RAG assistant ready. Type 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "salir"}:
            break
        try:
            response, chunks = answer(llm, vectorstore, question, k=k)
            print_answer(question, response, chunks)
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG call failed: %s", exc)
            print(f"[error] {exc}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Local RAG over the TechStream project documentation"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Single question to answer (omit with --interactive for REPL)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive REPL instead of a one-shot query",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the FAISS index even if a cache exists",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama HTTP endpoint (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"sentence-transformers checkpoint (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Where the FAISS index is persisted",
    )
    parser.add_argument(
        "--top-k", type=int, default=6, help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for ``python -m src.rag_assistant``."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    if not args.interactive and not args.question:
        print(
            "error: provide a question as a positional argument or use --interactive",
            file=sys.stderr,
        )
        return 2

    vectorstore = build_or_load_index(
        doc_paths=DEFAULT_DOC_PATHS,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        rebuild=args.rebuild,
    )

    llm = OllamaLLM(
        base_url=args.ollama_url,
        model=args.ollama_model,
        temperature=0.2,
    )

    try:
        if args.interactive:
            interactive_loop(llm, vectorstore, k=args.top_k)
        else:
            response, chunks = answer(
                llm, vectorstore, args.question, k=args.top_k
            )
            print_answer(args.question, response, chunks)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG pipeline failed: %s", exc)
        print(f"\n[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
