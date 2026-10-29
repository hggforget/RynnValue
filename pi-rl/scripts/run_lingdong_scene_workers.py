"""Run resumable local VLM annotation workers on the selected GPUs."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main(args):
    output = args.output.resolve()
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running, streams = {}, []
    try:
        for gpu in args.gpus:
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "USE_TF": "0",
                "USE_FLAX": "0",
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": "src",
            }
            stream = (logs / f"worker_{gpu}.log").open("a", buffering=1)
            streams.append(stream)
            command = [
                sys.executable,
                "-u",
                "scripts/infer_lingdong_scenes.py",
                "--output",
                str(output),
                "--worker-index",
                str(gpu),
                "--num-workers",
                "8",
                "--batch-size",
                str(args.batch_size),
            ]
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            running[gpu] = process
        (output / "worker_pids.json").write_text(
            json.dumps({str(gpu): p.pid for gpu, p in running.items()}, indent=2) + "\n"
        )
        previous = None
        while any(process.poll() is None for process in running.values()):
            counts = {}
            for gpu in running:
                path = output / f"worker_{gpu}_status.json"
                if path.is_file():
                    status = json.loads(path.read_text())
                    counts[gpu] = [status["done"], status["total"]]
            current = json.dumps(
                {
                    "workers": counts,
                    "predictions": len(list((output / "predictions").glob("episode_*.json"))),
                    "exit_codes": {gpu: p.poll() for gpu, p in running.items()},
                }
            )
            if current != previous:
                print(current, flush=True)
                previous = current
            time.sleep(5)
        result = {gpu: process.returncode for gpu, process in running.items()}
        print(json.dumps({"final_exit_codes": result}), flush=True)
        if any(result.values()):
            raise RuntimeError(f"Some VLM workers failed: {result}")
    except KeyboardInterrupt:
        for process in running.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        raise
    finally:
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("../output/lingdong_scene_annotations"))
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--batch-size", type=int, default=8)
    main(parser.parse_args())
