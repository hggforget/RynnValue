"""Check interpreter selection without launching a training job."""

# ruff: noqa: PT009
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


class PickplaceLaunchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pickplace launch ")
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        source = pathlib.Path(__file__).resolve().parents[3] / "scripts/train_sft_pku_franka_pickplace.sh"
        self.script = scripts / source.name
        shutil.copyfile(source, self.script)
        dataset = self.root / "dataset"
        (dataset / "meta").mkdir(parents=True)
        (dataset / "meta/info.json").write_text("{}")
        weights = self.root / "weights/params"
        weights.mkdir(parents=True)
        self.env = dict(os.environ)
        for key in (
            "PYTHON", "VIRTUAL_ENV", "CONDA_PREFIX", "DRY_RUN", "SKIP_NORM",
            "BATCH_SIZE", "BATCH_SIZE_PER_GPU", "FSDP_DEVICES",
        ):
            self.env.pop(key, None)
        self.env.update(REPO_ID=str(dataset), WEIGHT_PARAMS_PATH=str(weights), GPUS="0", DRY_RUN="1")

    def interpreter(self, prefix):
        path = self.root / prefix / "bin/python"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Dry runs must not invoke any interpreter.
        path.write_text("#!/bin/sh\nexit 99\n")
        path.chmod(0o755)
        return path

    def launch(self, **env):
        return subprocess.run(
            ["bash", str(self.script)], env={**self.env, **env},
            capture_output=True, text=True, check=False, timeout=10,
        )

    def assert_selected(self, expected, **env):
        result = self.launch(**env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Python: {expected}\n", result.stdout)

    def test_activated_conda_wins_over_checkout_venv(self):
        self.interpreter(".venv")
        active = self.interpreter("active conda")
        self.assert_selected(active, CONDA_PREFIX=str(active.parent.parent))

    def test_virtualenv_inside_conda_wins(self):
        conda = self.interpreter("conda")
        virtualenv = self.interpreter("virtual env")
        self.assert_selected(
            virtualenv, VIRTUAL_ENV=str(virtualenv.parent.parent), CONDA_PREFIX=str(conda.parent.parent),
        )

    def test_explicit_python_wins_over_activated_environments(self):
        active = self.interpreter("active")
        explicit = self.interpreter("explicit")
        self.assert_selected(
            explicit, PYTHON=str(explicit), VIRTUAL_ENV=str(active.parent.parent),
            CONDA_PREFIX=str(active.parent.parent),
        )

    def test_checkout_venv_is_used_without_an_activated_environment(self):
        self.assert_selected(self.interpreter(".venv"))

    def test_path_python_is_used_without_a_checkout_venv(self):
        path_python = self.interpreter("path env")
        self.assert_selected(path_python, PATH=f"{path_python.parent}{os.pathsep}{self.env['PATH']}")

    def fake_training_python(self, *, missing_numpy=False):
        path = self.interpreter("active conda")
        path.write_text(
            f"#!{sys.executable}\n"
            "import importlib.util, json, os, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "with open('calls.jsonl', 'a') as stream:\n"
            "    stream.write(json.dumps({'args': args, 'jax': os.getenv('JAX_PLATFORMS')}) + '\\n')\n"
            f"missing_numpy = {missing_numpy!r}\n"
            "if args[0] == '-':\n"
            "    importlib.util.find_spec = lambda name: None if missing_numpy and name == 'numpy' else object()\n"
            "    sys.argv = args\n"
            "    exec(compile(sys.stdin.read(), '<launcher-check>', 'exec'))\n"
            "elif args[0] == 'scripts/compute_norm_stats_fast.py':\n"
            "    path = pathlib.Path('assets/pi05_pku_franka_pickplace/pku_franka_pickplace/norm_stats.json')\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    stats = {key: {field: [0.0] * dim for field in ('mean', 'std', 'q01', 'q99')}\n"
            "             for key, dim in [('state', 14), ('actions', 7)]}\n"
            "    path.write_text(json.dumps({'norm_stats': stats}))\n"
            "elif args[0] != 'scripts/train.py':\n"
            "    raise SystemExit('Unexpected command: ' + repr(args))\n"
        )
        return path

    def test_missing_numpy_stops_before_norms_and_reports_the_interpreter(self):
        active = self.fake_training_python(missing_numpy=True)
        result = self.launch(CONDA_PREFIX=str(active.parent.parent), DRY_RUN="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing training dependencies", result.stderr)
        self.assertIn("numpy", result.stderr)
        self.assertIn(sys.executable, result.stderr)
        self.assertIn("PYTHON=/path/to/training/env/bin/python", result.stderr)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertEqual([call["args"] for call in calls], [["-"]])

    def test_same_active_python_runs_checks_norms_and_training(self):
        self.interpreter(".venv")
        active = self.fake_training_python()
        result = self.launch(CONDA_PREFIX=str(active.parent.parent), DRY_RUN="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Python: {active}\n", result.stdout)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[1]["args"][0], "scripts/compute_norm_stats_fast.py")
        self.assertEqual(calls[1]["jax"], "cpu")
        self.assertEqual(calls[3]["args"][:2], ["scripts/train.py", "pi05_pku_franka_pickplace"])


if __name__ == "__main__":
    unittest.main()
