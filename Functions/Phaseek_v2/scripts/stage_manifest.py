#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def staged_name(source: str) -> str:
    path = Path(source)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{digest}__{path.name}"


def copy_one(pair: tuple[str, Path]) -> tuple[str, str]:
    source, destination = pair
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size != Path(source).stat().st_size:
        temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    return source, str(destination.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage manifest NPZ files onto node-local storage")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    sources = sorted(set(frame["npz_path"].astype(str)))
    pairs = [(source, destination / staged_name(source)) for source in sources]
    mapping = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for source, staged in tqdm(executor.map(copy_one, pairs), total=len(pairs), desc="staging NPZ"):
            mapping[source] = staged
    frame["npz_path"] = frame["npz_path"].astype(str).map(mapping)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Staged manifest: {output}")


if __name__ == "__main__":
    main()
