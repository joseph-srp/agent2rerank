from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Pyserini/Lucene BM25 index from ESCI product JSONL docs.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing Pyserini JSONL files.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.index_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Index already exists: {args.index_dir}. Pass --overwrite to rebuild.")
        shutil.rmtree(args.index_dir)
    args.index_dir.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(args.input_dir),
        "--index",
        str(args.index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(args.threads),
        "--storeRaw",
    ]
    print(" ".join(command), flush=True)
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "sk-pyserini-lucene-only")
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()