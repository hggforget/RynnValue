"""Plot additional recorded gripper trajectories and current binary event labels."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from openpi.policies import lingdong_policy as policy

PREVIOUS_EXAMPLES = {0, 1, 128, 880, 1202, 1558}


def evenly_spaced(candidates: list[int], count: int) -> list[int]:
    if len(candidates) < count:
        raise ValueError(f"Need {count} distinct episodes, found {len(candidates)}")
    return [candidates[i] for i in np.linspace(0, len(candidates) - 1, count, dtype=int)]


def select_groups(rows: dict, metadata: dict, raw: dict) -> dict[str, list[int]]:
    used = set(PREVIOUS_EXAMPLES)
    repeated = sorted(i for i, row in rows.items() if row["new_switches"] >= 3 and i not in used)
    repeated = evenly_spaced(repeated, 12)
    used.update(repeated)
    initial_open = evenly_spaced(
        sorted(i for i, row in rows.items() if i not in used and row["new_switches"] == 1 and raw[i][0] >= 0.6), 4
    )
    used.update(initial_open)
    final_open = evenly_spaced(
        sorted(
            i for i, row in rows.items() if i not in used and row["new_switches"] == 1 and row["final_command"] == 1
        ),
        4,
    )
    used.update(final_open)
    terminal = sorted(
        (i for i, row in rows.items() if i not in used and row["events"]),
        key=lambda i: (rows[i]["frames"] - rows[i]["events"][-1]["confirmed_frame"], i),
    )[:4]
    used.update(terminal)
    ordinary = {}
    for task, name in (("pick up the object", "pick_up"), ("put down the object", "put_down")):
        ordinary[name] = evenly_spaced(
            sorted(
                i
                for i, row in rows.items()
                if i not in used and row["new_switches"] == 2 and task in metadata[i]["tasks"]
            ),
            12,
        )
        used.update(ordinary[name])
    return {**ordinary, "multiple_cycles": repeated, "boundaries": [*initial_open, *final_open, *terminal]}


def draw(axis, raw, labels, events, fps, title):
    time = np.arange(len(raw)) / fps
    axis.plot(time, raw, color="#64748b", linewidth=1.5, label="Recorded opening", zorder=3)
    axis.step(time, labels, where="post", color="#2563eb", linewidth=1.7, label="Binary action", zorder=4)
    for event in events:
        axis.axvline(
            event["frame"] / fps, color="#16a34a" if event["command"] else "#ea580c", alpha=0.35, linewidth=0.8
        )
    axis.set(title=title, xlim=(0, time[-1]), ylim=(-0.08, 1.1), yticks=[0, 0.5, 1], xlabel="Time (s)")
    axis.grid(alpha=0.15)
    axis.spines[["top", "right"]].set_visible(False)


def main(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    individual = output / "individual"
    individual.mkdir(exist_ok=True)
    rows = {row["episode_index"]: row for row in json.loads((source / "all_episode_events.json").read_text())}
    metadata = {row["episode_index"]: row for row in json.loads((source / "episodes.json").read_text())}
    with np.load(source / "raw_grippers.npz") as saved:
        raw = {int(name.split("_")[1]): saved[name] for name in saved.files}
    dataset = Path(json.loads((source / "manifest.json").read_text())["dataset"])
    fps = json.loads((dataset / "meta/info.json").read_text())["fps"]
    groups = select_groups(rows, metadata, raw)
    selected = [i for group in groups.values() for i in group]
    assert len(selected) == len(set(selected)) == 48
    assert not set(selected) & PREVIOUS_EXAMPLES
    titles = {
        "pick_up": "Pick-up: 12 additional episodes",
        "put_down": "Put-down: 12 additional episodes",
        "multiple_cycles": "Multiple open/close cycles: all 12 additional cases",
        "boundaries": "Boundaries: initially open, finally open, and late transitions",
    }
    manifest = {
        "source": str(source.resolve()),
        "detector": policy.gripper_binarization_spec(),
        "fps": fps,
        "groups": {},
    }
    with PdfPages(output / "gripper_curves_48_episodes.pdf") as pdf:
        for name, episodes in groups.items():
            fig, axes = plt.subplots(4, 3, figsize=(18, 12), constrained_layout=True)
            entries = []
            for axis, index in zip(axes.flat, episodes, strict=True):
                values = raw[index]
                labels, events = policy.detect_gripper_events(values)
                assert [(e["frame"], e["command"]) for e in events] == [
                    (e["frame"], e["command"]) for e in rows[index]["events"]
                ]
                task = "pick up" if "pick up the object" in metadata[index]["tasks"] else "put down"
                title = f"Ep {index:06d} | {task} | {len(events)} switches"
                draw(axis, values, labels, events, fps, title)
                single, single_axis = plt.subplots(figsize=(12, 3.4), constrained_layout=True)
                draw(single_axis, values, labels, events, fps, title)
                single_axis.set_ylabel("Opening / command")
                single_axis.legend(loc="upper right", fontsize=9)
                single.savefig(individual / f"episode_{index:06d}.png", dpi=160)
                plt.close(single)
                with (individual / f"episode_{index:06d}.csv").open("w") as file:
                    writer = csv.writer(file)
                    writer.writerow(["frame", "time_s", "recorded_opening", "binary_action"])
                    writer.writerows(
                        (t, t / fps, float(g), int(label))
                        for t, (g, label) in enumerate(zip(values, labels, strict=True))
                    )
                entries.append({"episode_index": index, "task": task, "frames": len(values), "events": events})
            fig.suptitle(titles[name] + "\nGray: recorded opening | Blue: binary action (0=close, 1=open)", fontsize=16)
            fig.savefig(output / f"{name}_12.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
            manifest["groups"][name] = entries
            print(f"Rendered {name}: {episodes}", flush=True)
    (output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../output/lingdong_gripper_event_review"))
    parser.add_argument("--output", type=Path, default=Path("../output/lingdong_gripper_more_curves"))
    args = parser.parse_args()
    main(args.source, args.output)
