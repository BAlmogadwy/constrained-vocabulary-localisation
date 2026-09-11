"""CLI for discounted batch VLM inference.

Supported batch providers: OpenAI, Anthropic, and Gemini. OpenRouter and
DashScope/Qwen are intentionally skipped here and should use run_vlm.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_environment  # noqa: E402

load_environment()

from experiments.vlm_batch import (  # noqa: E402
    collect_batch_results,
    download_batch_outputs,
    prepare_batches,
    status_batches,
    submit_batches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, submit, monitor, download, and collect discounted batch "
            "VLM inference for OpenAI, Anthropic, and Gemini models."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create provider batch inputs.")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--batch-name", default=None)
    prepare.add_argument("--test", action="store_true", help="Prepare only two images.")

    for name, help_text in (
        ("submit", "Submit prepared batches to provider APIs."),
        ("status", "Refresh provider batch status."),
        ("download", "Download completed batch output JSONL files."),
        ("collect", "Parse downloaded outputs into per-image caches and aggregates."),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--batch-root", required=True, type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = prepare_batches(args.config, batch_name=args.batch_name, test=args.test)
    elif args.command == "submit":
        result = submit_batches(args.batch_root)
    elif args.command == "status":
        result = status_batches(args.batch_root)
    elif args.command == "download":
        result = download_batch_outputs(args.batch_root)
    elif args.command == "collect":
        result = collect_batch_results(args.batch_root)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
