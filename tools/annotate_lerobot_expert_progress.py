"""Stage and verify dense LeRobot progress labels from a RynnValue R1 expert.

No source files are changed by inference. ``publish`` backs up and verifies every
affected source file before copying the staged parquet and metadata onto it.
Progress is negative predicted remaining seconds, as used by the FR3 RL adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def history_indices(episode, anchor, count=8, stride=8):
    # Match the training candidate stride, reserving the exact current frame.
    past = list(range(0, anchor, stride))
    rng = random.Random(episode * 1_000_003 + anchor)
    return sorted(rng.sample(past, min(count - 1, len(past)))) + [anchor]


def context(instruction, count):
    if instruction == "lace each object on the plate that matches its color.":
        instruction = "place each object on the plate that matches its color."
    return [{"role": "user", "content": [
        {"type": "text", "text": "The agent is a Franka single-arm robot."},
        {"type": "text", "text": f"The agent is performing the following task: {instruction}."},
        *[{"type": "image"} for _ in range(count)],
    ]}]


def add_columns(table, durations):
    import numpy as np
    import pyarrow as pa

    durations = np.asarray(durations, dtype=np.float32)
    assert durations.shape == (len(table),) and np.isfinite(durations).all()
    assert (durations >= 0).all()
    for key, values in [("remaining_time_s", durations), ("progress", -durations)]:
        if key in table.column_names:
            raise ValueError(f"Source already has {key}; refusing to overwrite existing labels")
        table = table.append_column(key, pa.array(values, type=pa.float32()))
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        hf = json.loads(metadata[b"huggingface"])
        for key in ("progress", "remaining_time_s"):
            hf["info"]["features"][key] = {"dtype": "float32", "_type": "Value"}
        metadata[b"huggingface"] = json.dumps(hf).encode()
    return table.replace_schema_metadata(metadata)


def worker(args):
    # Separate caches prevent concurrent transformers remote-code copy races.
    cache = args.work / "cache" / str(args.rank)
    os.environ["HF_MODULES_CACHE"] = str(cache / "modules")
    os.environ["HF_HOME"] = str(cache / "hf")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import av
    import numpy as np
    from PIL import Image
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    load_checkpoint = args.load_checkpoint or Path(os.environ.get("RYNNVALUE_LOAD_CHECKPOINT", str(args.checkpoint)))
    processor = AutoProcessor.from_pretrained(load_checkpoint, trust_remote_code=True)
    config = AutoConfig.from_pretrained(load_checkpoint, trust_remote_code=True)
    expert = config.model_type == "rynn_value_r1_with_expert"
    if not expert:
        assert config.model_type == "rynn_value_lang"
        config._attn_implementation = "pred_slot_isolated_eager"
    model = AutoModel.from_pretrained(
        load_checkpoint, config=config, trust_remote_code=True, dtype=torch.bfloat16,
        **({"attn_implementation": "sdpa"} if expert else {}),
    ).eval().requires_grad_(False).to(device="cuda", dtype=torch.bfloat16)
    if expert:
        assert model.config.value_representation == "temporal_distance"
    print(json.dumps({"event": "model_loaded", "rank": args.rank}), flush=True)
    info = json.loads((args.source / "meta/info.json").read_text())
    tasks = {r["task_index"]: r["task"] for r in read_jsonl(args.source / "meta/tasks.jsonl")}
    episodes = read_jsonl(args.source / "meta/episodes.jsonl")
    if args.episode_ids:
        selected = {int(e) for e in args.episode_ids.split(",")}
        episodes = [e for e in episodes if e["episode_index"] in selected]
        assert {e["episode_index"] for e in episodes} == selected
    if args.smoke:
        episodes = episodes[:1]
    else:
        episodes = episodes[args.rank::args.world_size]

    for ep in episodes:
        started = time.monotonic()
        eid = ep["episode_index"]
        rel = Path(info["data_path"].format(episode_chunk=eid // info["chunks_size"], episode_index=eid))
        out = args.work / "staged" / rel
        result_path = args.work / "episodes" / f"{eid:06d}.json"
        if not args.smoke and out.exists() and result_path.exists():
            assert digest(out) == json.loads(result_path.read_text())["output_sha256"]
            continue
        source_hash = digest(args.source / rel)
        table = pq.read_table(args.source / rel)
        n = len(table)
        assert n == ep["length"]
        frame_index = table["frame_index"].to_numpy()
        ts = table["timestamp"].to_numpy()
        assert np.array_equal(frame_index, np.arange(n))
        assert np.all(table["episode_index"].to_numpy() == eid)
        assert np.allclose(ts, frame_index / info["fps"], atol=2e-5)
        task_ids = np.unique(table["task_index"].to_numpy())
        assert len(task_ids) == 1
        instruction = args.instruction or tasks[int(task_ids[0])]

        def backbone_text(count):
            messages = context(instruction, count) + [{"role": "user", "content": [{
                "type": "text", "text": "Estimate the remaining time (in seconds) to complete the task."}]}]
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        video = args.source / info["video_path"].format(
            episode_chunk=eid // info["chunks_size"], episode_index=eid,
            video_key="observation.images.primary")
        images = []
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = 2
            assert float(stream.average_rate) == info["fps"]
            for i, frame in enumerate(container.decode(video=0)):
                assert abs(float(frame.time) - i / info["fps"]) < 2e-5
                images.append(frame.to_image().resize((320, 240), Image.Resampling.LANCZOS))
        assert len(images) == n, (eid, len(images), n)
        # Each image is transformed once; assembled histories have exactly the
        # same inputs as processor(text=..., images=history) for that frame.
        pixels = []
        grids = []
        for start in range(0, n, 32):
            item = processor.image_processor(images=images[start:start + 32], return_tensors="pt")
            grid = item["image_grid_thw"]
            assert torch.equal(grid, grid[:1].expand_as(grid))
            patches = int(grid[0].prod())
            pixels.append(item["pixel_values"].reshape(len(grid), patches, -1).to(torch.bfloat16))
            grids.append(grid)
        pixels = torch.cat(pixels)
        grids = torch.cat(grids)
        templates = {}
        for count in range(1, 9):
            if expert:
                text = (backbone_text(count) if args.backbone else
                        processor.apply_chat_template(context(instruction, count)))
                templates[count] = processor(text=text, images=[images[0]] * count, return_tensors="pt")
            else:
                templates[count] = processor.process_episode(
                    instruction=instruction, images=[images[0]] * count,
                    robot_description="a Franka single-arm robot")

        anchors = list(range(n))
        if args.smoke:
            anchors = sorted(set([0, 1, 7, 8, 16, 64, 100, n // 2, n - 2, n - 1]))
        durations = np.full(n, np.nan, dtype=np.float32)

        def assemble(selected):
            histories = [history_indices(eid, a) for a in selected]
            max_length = max(templates[len(h)]["input_ids"].shape[1] for h in histories)
            ids = torch.full((len(histories), max_length), processor.tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, history in enumerate(histories):
                item = templates[len(history)]
                size = item["input_ids"].shape[1]
                slot = slice(max_length - size, None) if args.backbone else slice(0, size)
                ids[j, slot] = item["input_ids"][0]
                mask[j, slot] = 1
            flat = [i for h in histories for i in h]
            return {"input_ids": ids.to("cuda"), "attention_mask": mask.to("cuda"),
                    "pixel_values": pixels[flat].flatten(0, 1).to("cuda"),
                    "image_grid_thw": grids[flat].to("cuda")}

        def predict(inputs):
            if args.backbone:
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                first = processor.tokenizer.convert_tokens_to_ids("<value_token_0>")
                bins = processor.value_tokenizer.n_bins
                eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
                allowed = torch.cat([logits[:, first:first + bins], logits[:, eos:eos + 1]], dim=1)
                selected = allowed.argmax(dim=1)
                assert (selected < bins).all(), "Backbone emitted end-of-turn instead of a value token"
                pred = processor.value_tokenizer.decode_index(selected).float()
            elif expert:
                pred = model.forward_value(**inputs).duration.float().reshape(-1)
            else:
                values = model(**inputs, use_cache=False).value.pred_value.float().mean(dim=0).reshape(-1)
                counts = inputs["input_ids"].eq(model.config.value_token_id).sum(dim=1)
                counts = counts // model.config.value_token_repeat
                assert int(counts.sum()) == len(values)
                pred = values[counts.cumsum(dim=0) - 1]
            return pred.cpu().numpy()

        with torch.inference_mode():
            for start in range(0, len(anchors), args.batch_size):
                chosen = anchors[start:start + args.batch_size]
                inputs = assemble(chosen)
                pred = predict(inputs)
                assert len(pred) == len(chosen) and np.isfinite(pred).all() and (pred >= 0).all()
                durations[chosen] = pred
                if (start // args.batch_size) % 25 == 0:
                    print(json.dumps({"event": "batch", "rank": args.rank, "episode": eid,
                                      "frames": start + len(chosen), "total": len(anchors),
                                      "seconds": round(time.monotonic() - started, 2)}), flush=True)
            if args.smoke:
                checks = []
                for anchor in anchors:
                    history = history_indices(eid, anchor)
                    if expert:
                        text = (backbone_text(len(history)) if args.backbone else
                                processor.apply_chat_template(context(instruction, len(history))))
                        direct = processor(text=text, images=[images[i] for i in history], return_tensors="pt")
                    else:
                        direct = processor.process_episode(
                            instruction=instruction, images=[images[i] for i in history],
                            robot_description="a Franka single-arm robot")
                        direct["pixel_values"] = direct["pixel_values"].flatten(0, 1)
                        direct["image_grid_thw"] = direct["image_grid_thw"].flatten(0, 1)
                    assembled = assemble([anchor])
                    for key in ("input_ids", "attention_mask", "image_grid_thw", "pixel_values"):
                        assert torch.equal(direct[key].to(assembled[key]), assembled[key]), (anchor, key)
                    single = float(predict(assembled)[0])
                    if args.backbone:
                        from transformers import LogitsProcessorList
                        first = processor.tokenizer.convert_tokens_to_ids("<value_token_0>")
                        bins = processor.value_tokenizer.n_bins
                        eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

                        class ValueOnly:
                            def __call__(self, input_ids, scores):
                                masked = torch.full_like(scores, float("-inf"))
                                masked[:, first:first + bins] = scores[:, first:first + bins]
                                masked[:, eos] = scores[:, eos]
                                return masked

                        generated = model.generate(
                            **assembled, max_new_tokens=1, do_sample=False, use_cache=True,
                            eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id,
                            logits_processor=LogitsProcessorList([ValueOnly()]))
                        native = float(processor.decode_value_tokens(generated[:, -1:]).reshape(-1)[0])
                        assert single == native, (anchor, single, native)
                    checks.append({"frame": anchor, "batch": float(durations[anchor]), "single": single,
                                   "absolute_difference_s": abs(single - float(durations[anchor]))})
                # BF16 kernels vary with padded batch shape. Use a relative
                # check across the different temporal scales of both models.
                if not args.backbone:
                    np.testing.assert_allclose(
                        [r["batch"] for r in checks], [r["single"] for r in checks],
                        rtol=0.02, atol=0.05)
                result = {"status": "SMOKE_PASSED", "rank": args.rank, "checks": checks,
                          "seconds": time.monotonic() - started,
                          "max_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
                write_json(args.work / "smoke.json", result)
                print(json.dumps(result), flush=True)
                return

        label_source = table
        if args.backbone:
            label_source = table.drop_columns([c for c in ("progress", "remaining_time_s") if c in table.column_names])
        labeled = add_columns(label_source, durations)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.parquet")
        pq.write_table(labeled, tmp, compression="snappy")
        tmp.replace(out)
        result = {"episode_index": eid, "frames": n, "relative_path": str(rel),
                  "instruction": instruction, "model_type": config.model_type,
                  "readout": "backbone_greedy_value_token" if args.backbone else "value_head",
                  "source_sha256": source_hash, "output_sha256": digest(out),
                  "first_remaining_time_s": float(durations[0]),
                  "last_remaining_time_s": float(durations[-1]),
                  "min_remaining_time_s": float(durations.min()),
                  "max_remaining_time_s": float(durations.max()),
                  "seconds": round(time.monotonic() - started, 3)}
        write_json(result_path, result)
        print(json.dumps({"event": "episode_complete", "rank": args.rank, **result}), flush=True)


def stage_metadata(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = json.loads((args.source / "meta/info.json").read_text())
    all_results = [json.loads(p.read_text()) for p in sorted((args.work / "episodes").glob("*.json"))]
    expected_ids = (sorted(int(e) for e in args.episode_ids.split(",")) if args.episode_ids
                    else list(range(info["total_episodes"])))
    assert [r["episode_index"] for r in all_results] == expected_ids
    if args.instruction:
        assert all(r.get("instruction") == args.instruction for r in all_results)
    stats = {r["episode_index"]: r for r in read_jsonl(args.source / "meta/episodes_stats.jsonl")}
    files = []
    frames = 0
    for result in all_results:
        rel = result["relative_path"]
        assert digest(args.source / rel) == result["source_sha256"]
        assert digest(args.work / "staged" / rel) == result["output_sha256"]
        original = pq.read_table(args.source / rel)
        labeled = pq.read_table(args.work / "staged" / rel)
        unchanged = [c for c in original.column_names if not (args.backbone and c in ("progress", "remaining_time_s"))]
        assert labeled.select(unchanged).equals(original.select(unchanged), check_metadata=False), rel
        assert len(labeled) == result["frames"]
        p = labeled["progress"].to_numpy()
        d = labeled["remaining_time_s"].to_numpy()
        assert np.isfinite(p).all() and np.isfinite(d).all() and np.array_equal(p, -d)
        for key in ("progress", "remaining_time_s"):
            assert labeled.schema.field(key).type == pa.float32()
            v = labeled[key].to_numpy().astype(np.float64)
            stats[result["episode_index"]]["stats"][key] = {
                "min": [float(v.min())], "max": [float(v.max())],
                "mean": [float(v.mean())], "std": [float(v.std())], "count": [len(v)]}
        frames += len(labeled)
        files.append({"path": rel, "source_sha256": result["source_sha256"],
                      "output_sha256": result["output_sha256"]})
    if args.episode_ids:
        assert frames <= info["total_frames"]
    else:
        assert frames == info["total_frames"]
    for key in ("progress", "remaining_time_s"):
        assert args.episode_ids or key not in info["features"]
        info["features"][key] = {"dtype": "float32", "shape": [1], "names": None}
    model_config = json.loads((args.checkpoint / "config.json").read_text())
    export_manifest_path = args.checkpoint / "export_manifest.json"
    model_hash = (json.loads(export_manifest_path.read_text())["export_files"]["model.safetensors"]["sha256"]
                  if export_manifest_path.exists() else None)
    annotation = {"checkpoint": str(args.checkpoint), "source": str(args.source),
                  "model_type": model_config["model_type"],
                  "instruction_override": args.instruction,
                  "readout": "backbone_greedy_value_token" if args.backbone else "value_head",
                  "progress_formula": "progress = -remaining_time_s",
                  "remaining_time_units": "seconds", "higher_progress_is_better": True,
                  "camera": "observation.images.primary", "history_frames": 8,
                  "history_sampling": "random past stride-8 candidates, exact current anchor included",
                  "seed": "episode_index * 1000003 + frame_index",
                  "inference_stride": 1, "interpolation": False, "future_observations": False,
                  "resize": [320, 240], "robot_description": "a Franka single-arm robot",
                  "instruction_correction": "lace each object -> place each object (inference only)",
                  "precision": "bfloat16", "attention": (
                      "pred_slot_isolated_eager" if model_config["model_type"] == "rynn_value_lang" else "sdpa"),
                  "batch_size": args.batch_size,
                  "terminal_predictions_forced": False,
                  "model_safetensors_sha256": model_hash,
                  "episodes": len(all_results), "frames": frames}
    info["progress_annotation"] = annotation
    for rel in ("meta/info.json", "meta/episodes_stats.jsonl"):
        target = args.work / "staged" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("info.json"):
            write_json(target, info)
        else:
            target.write_text("".join(json.dumps(stats[e], ensure_ascii=False) + "\n" for e in sorted(stats)))
        files.append({"path": rel, "source_sha256": digest(args.source / rel), "output_sha256": digest(target)})
    manifest = {"status": "VERIFIED_SAMPLE" if args.episode_ids else "VERIFIED_STAGED",
                **annotation, "files": files, "episode_results": all_results}
    write_json(args.work / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "episodes": len(all_results), "frames": frames}), flush=True)


def run(args):
    args.work.mkdir(parents=True, exist_ok=True)
    logdir = args.work / "logs"
    logdir.mkdir(exist_ok=True)
    procs = []
    for rank, gpu in enumerate(args.gpus.split(",")):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        log = open(logdir / f"worker_{rank}.log", "a", buffering=1)
        command = [sys.executable, str(Path(__file__).resolve()), "worker",
                   "--source", str(args.source), "--checkpoint", str(args.checkpoint),
                   "--work", str(args.work), "--rank", str(rank),
                   "--world-size", str(len(args.gpus.split(","))), "--batch-size", str(args.batch_size)]
        if args.instruction:
            command.extend(["--instruction", args.instruction])
        if args.episode_ids:
            command.extend(["--episode-ids", args.episode_ids])
        if args.load_checkpoint:
            command.extend(["--load-checkpoint", str(args.load_checkpoint)])
        if args.backbone:
            command.append("--backbone")
        procs.append((rank, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    failed = []
    while procs:
        remaining = []
        for rank, proc, log in procs:
            status = proc.poll()
            if status is None:
                remaining.append((rank, proc, log))
            else:
                log.close()
                print(json.dumps({"event": "worker_exit", "rank": rank, "exit_code": status}), flush=True)
                if status:
                    failed.append(rank)
        procs = remaining
        if procs:
            time.sleep(5)
    if failed:
        raise RuntimeError(f"Workers failed: {failed}; inspect {logdir}")
    stage_metadata(args)


def publish_previews(args):
    target = args.source / "annotations" / args.annotation_name
    previews = args.work / "previews"
    records = []
    for source in sorted(previews.iterdir()):
        if source.suffix not in (".mp4", ".png", ".jpg"):
            continue
        dest = target / "previews" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected = digest(source)
        if dest.exists():
            assert digest(dest) == expected, f"Existing preview differs: {dest}"
        else:
            shutil.copyfile(source, dest)
        assert digest(dest) == expected
        records.append({"path": str(dest.relative_to(target)), "bytes": dest.stat().st_size, "sha256": expected})
    (target / "preview_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    local_manifest = args.work / "manifest.json"
    if local_manifest.exists() and json.loads(local_manifest.read_text())["status"] == "VERIFIED_SAMPLE":
        shutil.copyfile(local_manifest, target / "evaluation_manifest.json")
        for filename in ("smoke.json", "model_comparison.json"):
            if (args.work / filename).exists():
                shutil.copyfile(args.work / filename, target / filename)
    print(json.dumps({"previews": str(target / "previews"), "files": len(records)}), flush=True)
    return records


def publish(args):
    manifest = json.loads((args.work / "manifest.json").read_text())
    assert manifest["source"] == str(args.source)
    assert manifest["status"] == "VERIFIED_STAGED"
    backup = args.source / "annotations" / args.annotation_name
    # Complete preflight and backups precede the first dataset mutation.
    for item in manifest["files"]:
        rel = item["path"]
        assert digest(args.work / "staged" / rel) == item["output_sha256"], rel
        current = digest(args.source / rel)
        assert current in (item["source_sha256"], item["output_sha256"]), f"Source changed: {rel}"
        saved = backup / "original" / rel
        if saved.exists():
            assert digest(saved) == item["source_sha256"], f"Invalid backup: {rel}"
        else:
            assert current == item["source_sha256"], f"Missing original backup: {rel}"
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.source / rel, saved)
            assert digest(saved) == item["source_sha256"]
    for item in manifest["files"]:
        rel = item["path"]
        assert digest(args.source / rel) in (item["source_sha256"], item["output_sha256"]), rel
        shutil.copyfile(args.work / "staged" / rel, args.source / rel)
        assert digest(args.source / rel) == item["output_sha256"], rel
    manifest["preview_files"] = publish_previews(args)
    manifest["status"] = "COMPLETE"
    manifest["backup_directory"] = str(backup / "original")
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # Plain sequential writes are supported by the object-store mount.
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    shutil.copyfile(__file__, backup / Path(__file__).name)
    write_json(args.work / "published.json", manifest)
    print(json.dumps({"status": "COMPLETE", "episodes": manifest["episodes"],
                      "frames": manifest["frames"], "backup": str(backup)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["worker", "run", "verify", "publish", "publish-previews"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--load-checkpoint", type=Path, default=None, help="Identical local weight cache; lineage keeps --checkpoint.")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--episode-ids", default=None, help="Optional sample subset; cannot publish dataset labels.")
    parser.add_argument("--annotation-name", default="progress_expert_sup_20261018_015318_step005000_20261025")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--backbone", action="store_true", help="Evaluate the expert checkpoint's native LM value tokens.")
    args = parser.parse_args()
    if args.backbone and args.mode in ("run", "worker", "verify"):
        assert args.smoke or args.episode_ids, "Backbone evaluation requires an explicit sample subset"
    {"worker": worker, "run": run, "verify": stage_metadata, "publish": publish,
     "publish-previews": publish_previews}[args.mode](args)


if __name__ == "__main__":
    main()
