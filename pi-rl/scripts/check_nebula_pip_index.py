"""Check the worker's package index before pip spends minutes retrying it."""

import os
import sys
import urllib.parse
import urllib.request


def main():
    index = os.environ["PIP_INDEX_URL"]
    parsed = urllib.parse.urlsplit(index)
    display = urllib.parse.urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))
    print(f"Checking worker package index: {display}", flush=True)
    try:
        url = urllib.parse.urljoin(index.rstrip("/") + "/", "jax/")
        with urllib.request.urlopen(url, timeout=5) as response:
            page = response.read(2 * 1024 * 1024)
        if b"jax-0.5.3-" not in page:
            raise RuntimeError("The index did not list jax==0.5.3")
    except Exception as error:
        print(
            f"Worker package-index check failed ({type(error).__name__}). "
            "Set a PIP_INDEX_URL reachable from the worker, or submit with "
            "--wheelhouse pointing to a complete wheel bundle on its OSS mount.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print("Worker package index lists jax==0.5.3", flush=True)


if __name__ == "__main__":
    main()
