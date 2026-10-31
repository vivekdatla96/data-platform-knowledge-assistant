"""Command line interface: ``dpka ingest | ask | eval | stats | serve``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dpka.config import Settings


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env(
        corpus_dir=getattr(args, "corpus", None),
        index_path=getattr(args, "index", None),
        embedding_backend=getattr(args, "embedder", None),
        retrieval_mode=getattr(args, "mode", None),
        llm_provider=getattr(args, "provider", None),
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    from dpka.embeddings import build_embedder
    from dpka.ingest import ingest
    from dpka.store import VectorStore

    s = _settings(args)
    store = VectorStore(s.index_path)
    report = ingest(
        s.corpus_dir,
        store,
        build_embedder(s.embedding_backend, s.embedding_model),
        rebuild=args.rebuild,
    )
    print(f"Ingested {s.corpus_dir} -> {s.index_path}")
    print(report.summary())
    for label, items in (
        ("added", report.added),
        ("updated", report.updated),
        ("deleted", report.deleted),
    ):
        for doc in items:
            print(f"  {label:8} {doc}")
    for doc, err in report.failed.items():
        print(f"  FAILED   {doc}: {err}")
    print(json.dumps(store.stats()))
    return 1 if report.failed and args.strict else 0


def _print_result(res: dict) -> None:
    print(f"\nQ: {res['question']}")
    print(f"Mode: {res['provider_label']} | retrieval: {res['retrieval_mode']}")
    if res["status"] == "answered":
        print(f"\n{res['answer']}\n\nSources:")
        for c in res["citations"]:
            lines = f" lines {c['start_line']}-{c['end_line']}" if c["start_line"] else ""
            print(f"  [{c['label']}] {c['doc_id']} :: {c['locator']}{lines}")
    elif res["status"] == "abstained":
        print(f"\nABSTAINED: {res['abstain_reason']}")
    else:
        print("\nGENERATION FAILED - retrieved evidence follows.")
    for w in res["warnings"]:
        print(f"  ! {w}")
    print("\nRetrieved evidence:")
    for e in res["evidence"]:
        flag = "  [untrusted: instruction-like text]" if e["suspicious"] else ""
        print(
            f"  {e['label']} sim={e['dense_score']:.3f} bm25={e['bm25_score']:.2f} "
            f"{e['doc_id']} :: {e['locator']}{flag}"
        )


def cmd_ask(args: argparse.Namespace) -> int:
    from dpka.assistant import Assistant

    assistant = Assistant.from_settings(_settings(args))
    res = assistant.ask(" ".join(args.question), top_k=args.k).to_dict()
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        _print_result(res)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from dpka.assistant import Assistant
    from dpka.evaluation import evaluate, load_questions, to_markdown

    assistant = Assistant.from_settings(_settings(args))
    report = evaluate(assistant, load_questions(Path(args.questions)), k=args.k)
    md = to_markdown(report)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        out.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
        print(f"\nWrote {out} and {out.with_suffix('.md')}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from dpka.store import VectorStore

    print(json.dumps(VectorStore(_settings(args).index_path).stats(), indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from dpka.api import create_app

    uvicorn.run(create_app(_settings(args)), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dpka", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index", type=Path, help="SQLite index path (DPKA_INDEX_PATH)")
    common.add_argument(
        "--embedder",
        choices=["fastembed", "hashing"],
        help="embedding backend (DPKA_EMBEDDING_BACKEND)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", parents=[common], help="index the corpus (incremental)")
    s.add_argument("--corpus", type=Path, help="corpus directory (DPKA_CORPUS_DIR)")
    s.add_argument("--rebuild", action="store_true", help="drop the index and rebuild")
    s.add_argument("--strict", action="store_true", help="exit 1 if any document fails")
    s.set_defaults(func=cmd_ingest)

    qa = argparse.ArgumentParser(add_help=False, parents=[common])
    qa.add_argument("--mode", choices=["dense", "hybrid"], help="retrieval mode")
    qa.add_argument("--provider", choices=["offline", "bedrock"], help="answer provider")
    qa.add_argument("-k", type=int, default=5, help="evidence chunks to retrieve")

    s = sub.add_parser("ask", parents=[qa], help="answer one question")
    s.add_argument("question", nargs="+")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("eval", parents=[qa], help="run the labeled evaluation set")
    s.add_argument("--questions", default="data/eval/questions.jsonl")
    s.add_argument("--out", help="write JSON (and .md) results here")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("stats", parents=[common], help="show index statistics")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("serve", parents=[qa], help="run the web UI and JSON API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
