#!/usr/bin/env python3
"""
Download a model file from Hugging Face Hub using ``huggingface-cli``.

Usage (defaults to bartowski/Qwen2.5-1.5B-Instruct-GGUF Q4_K_M into ``models/``):
  python scripts/download_model.py

Override examples:
  python scripts/download_model.py \
    --repo bartowski/Qwen2.5-1.5B-Instruct-GGUF \
    --file Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
    --out models
"""
import argparse
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="bartowski/Qwen2.5-1.5B-Instruct-GGUF")
    p.add_argument("--file", default="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
    p.add_argument("--out", default="models")
    args = p.parse_args()

    dest_dir = Path(args.out)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo}::{args.file} -> {dest_dir}")
    cmd = [
        "huggingface-cli",
        "download",
        args.repo,
        args.file,
        "--local-dir",
        str(dest_dir),
    ]
    try:
        res = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        print(res.stdout)
    except FileNotFoundError:
        print("huggingface-cli not found. Install via: pip install -U huggingface-hub")
        return
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        raise SystemExit(e.returncode)
    print(f"Saved to {dest_dir / args.file}")


if __name__ == "__main__":
    main()

