#!/usr/bin/env python3
"""Download a model file from Hugging Face Hub using the Python API.

By default this grabs the Qwen2.5 1.5B Instruct GGUF Q4_K_M model into
``models/``.  Use --repo/--file/--out to override.
"""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="bartowski/Qwen2.5-1.5B-Instruct-GGUF")
    p.add_argument("--file", default="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
    p.add_argument("--out", default="models")
    args = p.parse_args()

    dest_dir = Path(args.out)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo}::{args.file} -> {dest_dir}")
    hf_hub_download(args.repo, args.file, local_dir=str(dest_dir), local_dir_use_symlinks=False)
    print(f"Saved to {dest_dir / args.file}")


if __name__ == "__main__":
    main()
