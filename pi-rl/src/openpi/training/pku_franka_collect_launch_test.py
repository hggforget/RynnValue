"""Exercise the offline launcher and its real norm validator without GPU dependencies."""

# ruff: noqa: PT009
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


class CollectLaunchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fr3 collect launch ")
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        source = pathlib.Path(__file__).resolve().parents[3] / "scripts/train_pi05_pku_franka_collect_objects.sh"
        self.script = scripts / source.name
        shutil.copyfile(source, self.script)
        self.dataset = self.root / "dataset"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "meta/info.json").write_text("{}")
        self.python = self.root / "fake-python"
        self.python.write_text(
            f"#!{sys.executable}\n"
            "import json, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "with open('calls.jsonl', 'a') as stream:\n"
            "    stream.write(json.dumps(args) + '\\n')\n"
            "if args[0] == '-':\n"
            "    sys.argv = args\n"
            "    exec(compile(sys.stdin.read(), '<norm-validator>', 'exec'))\n"
            "elif args[0] == 'scripts/compute_norm_stats_fast.py':\n"
            "    config = args[args.index('--config-name') + 1]\n"
            "    asset = 'pku_franka_collect_objects'\n"
            "    if config.startswith('pi05_pku_franka_collect_objects_relative'):\n"
            "        asset += '_relative'\n"
            "    path = pathlib.Path('assets') / config / asset / 'norm_stats.json'\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    stats = {key: {field: [0.0] * dim for field in ('mean', 'std', 'q01', 'q99')}\n"
            "             for key, dim in [('state', 14), ('actions', 7)]}\n"
            "    path.write_text(json.dumps({'norm_stats': stats}))\n"
            "elif args[0] not in ('scripts/train.py', 'scripts/train_iql.py'):\n"
            "    raise SystemExit('Unexpected command: ' + repr(args))\n"
        )
        self.python.chmod(0o755)

    def launch(self, *, method="sft", representation=None, config=None, skip_norm=False):
        env = dict(os.environ)
        for key in ["METHOD", "CONFIG", "ACTION_REPRESENTATION", "SKIP_NORM", "INIT_CHECKPOINT"]:
            env.pop(key, None)
        env.update(PYTHON=str(self.python), REPO_ID=str(self.dataset), METHOD=method)
        if representation is not None:
            env["ACTION_REPRESENTATION"] = representation
        if config is not None:
            env["CONFIG"] = config
        if skip_norm:
            env["SKIP_NORM"] = "1"
        return subprocess.run(
            ["bash", str(self.script)], env=env, capture_output=True, text=True, check=False, timeout=10
        )

    def test_all_eight_presets_route_to_matching_trainer_and_norm_assets(self):
        for representation in ["absolute", "relative"]:
            prefix = "pi05_pku_franka_collect_objects" + ("_relative" if representation == "relative" else "")
            for method, suffix in [("sft", ""), ("sft", "_lora"), ("iql", "_iql"), ("filter_bc", "_filter_bc")]:
                with self.subTest(representation=representation, method=method, suffix=suffix):
                    config = prefix + suffix
                    log = self.root / "calls.jsonl"
                    log.unlink(missing_ok=True)
                    # Explicit LoRA CONFIG must also work without an action env var.
                    result = self.launch(
                        method=method,
                        representation=None if suffix == "_lora" else representation,
                        config=config if suffix == "_lora" else None,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    calls = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertEqual(calls[0][calls[0].index("--config-name") + 1], config)
                    asset = "pku_franka_collect_objects" + ("_relative" if representation == "relative" else "")
                    self.assertEqual(calls[1], ["-", f"./assets/{config}/{asset}/norm_stats.json"])
                    trainer = "scripts/train_iql.py" if method == "iql" else "scripts/train.py"
                    self.assertEqual(calls[2][:2], [trainer, config])

    def test_conflicting_or_invalid_modes_fail_before_training(self):
        for kwargs in [
            {"representation": "invalid"},
            {"representation": "relative", "config": "pi05_pku_franka_collect_objects"},
            {"representation": "absolute", "config": "pi05_pku_franka_collect_objects_relative"},
            {"method": "iql", "config": "pi05_pku_franka_collect_objects_relative"},
        ]:
            with self.subTest(kwargs=kwargs):
                result = self.launch(**kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "calls.jsonl").exists())

    def test_relative_skip_norm_never_loads_absolute_asset_directory(self):
        config = "pi05_pku_franka_collect_objects_relative"
        path = self.root / "assets" / config / "pku_franka_collect_objects/norm_stats.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        result = self.launch(representation="relative", skip_norm=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pku_franka_collect_objects_relative/norm_stats.json", result.stderr)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), 1)  # Only the validator ran; no trainer was launched.


if __name__ == "__main__":
    unittest.main()
