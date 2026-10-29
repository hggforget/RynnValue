"""Stage the requested NAS model with streaming I/O and per-file SHA256 receipts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import hashlib
import json
from pathlib import Path
import time


def copy_one(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    checksum = hashlib.sha256()
    length = 0
    with source.open("rb") as src, partial.open("wb") as dst:
        while chunk := src.read(16 * 1024 * 1024):
            dst.write(chunk)
            checksum.update(chunk)
            length += len(chunk)
    if length != source.stat().st_size:
        raise ValueError(f"Model file changed during staging: {source.name}")
    partial.replace(destination)
    return {
        "name": source.name,
        "bytes": length,
        "sha256": checksum.hexdigest(),
        "source_mtime_ns": source.stat().st_mtime_ns,
    }


def main(source, destination, workers):
    files = sorted(path for path in source.iterdir() if path.is_file())
    started = time.monotonic()
    receipts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_one, path, destination / path.name) for path in files]
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            print(f"Staged {receipt['name']}: {receipt['bytes'] / 2**30:.2f} GiB", flush=True)
    (destination / "staging_receipt.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "files": receipts,
                "elapsed_s": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Complete: {sum(r['bytes'] for r in receipts) / 2**30:.2f} GiB in {time.monotonic() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/mnt/rynnbot_hangzhou_extreme/CKPT/PUBLIC/Qwen3-VL-30B-A3B-Instruct")
    )
    parser.add_argument("--destination", type=Path, default=Path("/tmp/lingdong_qwen3_vl_30b_a3b_cache"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(args.source, args.destination, args.workers)
