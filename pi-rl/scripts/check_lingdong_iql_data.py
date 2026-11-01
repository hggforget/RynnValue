"""Validate Lingdong IQL reward coverage without decoding videos or loading weights."""

import argparse
import json
from pathlib import Path

from openpi.policies import lingdong_language
from openpi.policies import lingdong_policy as policy
from openpi.policies import lingdong_rl_policy as rl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(policy.DEFAULT_DATASET_ROOT))
    parser.add_argument("--task", choices=("all", "pick_up", "put_down"), default="all")
    parser.add_argument("--reward-source", choices=("progress", "terminal"), default="progress")
    parser.add_argument("--reward-labels-path")
    parser.add_argument("--assume-success", action="store_true")
    parser.add_argument("--print-dataset-signature", action="store_true")
    args = parser.parse_args()
    repo = str(args.dataset_root / policy.DATASET_NAMES[0])
    if args.print_dataset_signature:
        print(lingdong_language.dataset_signature(Path(repo)))
        return
    metadata = rl.reward_metadata(
        repo,
        task=args.task,
        reward_source=args.reward_source,
        reward_labels_path=args.reward_labels_path,
        assume_success=args.assume_success,
    )
    print(
        json.dumps(
            {
                "status": "valid",
                "dataset": repo,
                "task": args.task,
                "episodes": len(metadata["episode_ids"]),
                "frames": sum(metadata["episode_lengths"][index] for index in metadata["episode_ids"]),
                "reward_source": args.reward_source,
                "progress_source": "dataset"
                if metadata["dataset_progress"]
                else "sidecar"
                if metadata["progress"]
                else None,
                "labeled_outcomes": len(metadata["outcomes"]),
                "assume_success": args.assume_success,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
