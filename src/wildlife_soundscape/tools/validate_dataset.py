"""Command-line validation for native research dataset manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildlife_soundscape.datasets import validate_dataset


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a three-node Wildlife Soundscape dataset manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Permit synthetic datasets for software-regression validation.",
    )
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = validate_dataset(
        args.manifest,
        scientific=not args.allow_synthetic,
        verify_audio=not args.skip_audio,
        verify_checksums=args.verify_checksums,
    )
    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        state = "VALID" if report.valid else "INVALID"
        print(
            f"{state}: {report.records_checked} records, "
            f"{report.audio_files_checked} audio files checked"
        )
        for result in report.issues:
            scope = f" [{result.record_id}]" if result.record_id else ""
            print(f"{result.severity.upper()} {result.code}{scope}: {result.message}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
