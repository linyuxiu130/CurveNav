#!/usr/bin/env python3
"""Resume a large gated Hugging Face archive with disjoint byte ranges."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import time

import requests


def resolve(url: str, token: str) -> str:
    response = requests.get(
        url, headers={"Authorization": f"Bearer {token}"},
        allow_redirects=False, timeout=60,
    )
    response.raise_for_status()
    return response.headers["Location"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("expected_size", type=int)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
        token = token_path.read_text().strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(args.output.name + ".chunks")
    staging.mkdir(exist_ok=True)
    ranges = [
        (start, min(start + args.chunk_size, args.expected_size) - 1)
        for start in range(0, args.expected_size, args.chunk_size)
    ]

    def fetch(item: tuple[int, int]) -> None:
        start, end = item
        part = staging / f"{start:016d}-{end:016d}"
        if part.exists() and part.stat().st_size == end - start + 1:
            return
        for attempt in range(20):
            try:
                signed = resolve(args.url, token)
                with requests.get(
                    signed, headers={"Range": f"bytes={start}-{end}"},
                    stream=True, timeout=(60, 300),
                ) as response:
                    response.raise_for_status()
                    if response.headers.get("Content-Range", "").split("/")[0] != f"bytes {start}-{end}":
                        raise RuntimeError(f"server ignored range {start}-{end}")
                    temporary = part.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(8 * 1024 * 1024):
                            stream.write(block)
                    if temporary.stat().st_size != end - start + 1:
                        raise RuntimeError(f"short range {start}-{end}")
                    temporary.replace(part)
                    return
            except (requests.RequestException, RuntimeError):
                if attempt == 19:
                    raise
                time.sleep(min(2 ** min(attempt, 5), 30))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, item) for item in ranges]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index == 1 or index % 32 == 0 or index == len(ranges):
                print(f"[download] {index}/{len(ranges)} ranges", flush=True)
    temporary = args.output.with_suffix(args.output.suffix + ".assembling")
    with temporary.open("wb") as destination:
        for start, end in ranges:
            destination.write((staging / f"{start:016d}-{end:016d}").read_bytes())
    if temporary.stat().st_size != args.expected_size:
        raise RuntimeError("assembled archive has the wrong size")
    temporary.replace(args.output)
    for part in staging.iterdir():
        part.unlink()
    staging.rmdir()


if __name__ == "__main__":
    main()
