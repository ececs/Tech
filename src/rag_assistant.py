"""CLI wrapper over the shared TechStream RAG service."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from pathlib import Path

from src.rag_service import (
    DEFAULT_DOC_PATHS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    ask_question,
    load_rag_resources,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)


def print_answer(question: str, response: str, chunks: list[object]) -> None:
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
        print(
            f"  [{i}] {chunk.source}:{chunk.start_line}  (score={chunk.score:.3f})"
        )
        print(f"      {preview}")
    print()


def interactive_loop(resources: object) -> None:
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
            response, chunks = ask_question(question, resources)
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
        "--fallback-model",
        type=str,
        default=DEFAULT_FALLBACK_MODEL,
        help=(
            "Local fallback model to try if the remote Ollama server is unreachable "
            f"(default: {DEFAULT_FALLBACK_MODEL})"
        ),
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

    resources = load_rag_resources(
        doc_paths=DEFAULT_DOC_PATHS,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        fallback_model=args.fallback_model,
        top_k=args.top_k,
        rebuild=args.rebuild,
    )
    if not resources.ready:
        print(f"\n[error] {resources.error}", file=sys.stderr)
        return 1

    try:
        if args.interactive:
            interactive_loop(resources)
        else:
            response, chunks = ask_question(args.question, resources)
            print_answer(args.question, response, chunks)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG pipeline failed: %s", exc)
        print(f"\n[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
