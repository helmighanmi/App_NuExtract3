#!/usr/bin/env python3
"""Download NuExtract3 weights for offline / air-gapped serving."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="numind/NuExtract3")
    parser.add_argument("--local-dir", type=Path, default=Path("models/NuExtract3"))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required. Install it with: uv pip install huggingface-hub"
        ) from None

    args.local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        revision=args.revision,
        token=args.token,
        max_workers=8,
    )
    print(f"Downloaded {args.repo_id} to {path}")


if __name__ == "__main__":
    main()
