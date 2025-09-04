#!/usr/bin/env python3
"""
Download a model file from Hugging Face Hub into a local directory.

Usage (defaults to bartowski/Qwen2.5-1.5B-Instruct-GGUF Q4_K_M into models/):
  python scripts/download_model.py

Override examples:
  python scripts/download_model.py \
    --repo bartowski/Qwen2.5-1.5B-Instruct-GGUF \
    --file Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
    --out models
"""
import argparse
from pathlib import Path
import shutil

try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    print("huggingface_hub is required. Install with: pip install -U huggingface-hub")
    raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="bartowski/Qwen2.5-1.5B-Instruct-GGUF")
    p.add_argument("--file", default="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
    p.add_argument("--out", default="models")
    args = p.parse_args()

    dest_dir = Path(args.out)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo}::{args.file} -> {dest_dir}")
    local = hf_hub_download(repo_id=args.repo, filename=args.file)
    target = dest_dir / args.file
    shutil.copy2(local, target)
    print(f"Saved to {target}")


if __name__ == "__main__":
    main()

