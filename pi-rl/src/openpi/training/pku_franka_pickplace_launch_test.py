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
            "PYTHON", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_EXE", "DRY_RUN", "SKIP_NORM",
            "BATCH_SIZE", "BATCH_SIZE_PER_GPU", "FSDP_DEVICES",
        ):
            self.env.pop(key, None)
        self.env.update(REPO_ID=str(dataset), WEIGHT_PARAMS_PATH=str(weights), GPUS="0", DRY_RUN="1")

    def interpreter(self, prefix, *, missing_numpy=False, incompatible_lerobot=False):
        path = self.root / prefix / "bin/python"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"#!{sys.executable}\n"
            "import importlib.util, json, os, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "with open('calls.jsonl', 'a') as stream:\n"
            "    stream.write(json.dumps({'args': args, 'jax': os.getenv('JAX_PLATFORMS')}) + '\\n')\n"
            f"missing_numpy = {missing_numpy!r}\n"
            f"incompatible_lerobot = {incompatible_lerobot!r}\n"
            "if args[0] == '-':\n"
            "    def find_spec(name):\n"
            "        if (missing_numpy and name == 'numpy') or (incompatible_lerobot and name.startswith('lerobot.common')):\n"
            "            return None\n"
            "        return object()\n"
            "    importlib.util.find_spec = find_spec\n"
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

    def test_no_activation_discovers_named_conda_after_rejecting_path_python(self):
        path_python = self.interpreter("path env", missing_numpy=True)
        shutil.copyfile(path_python, path_python.with_name("python3"))
        path_python.with_name("python3").chmod(0o755)
        discovered = self.interpreter("conda installation/envs/pi-rl")
        self.assert_selected(
            discovered, CONDA_EXE=str(self.root / "conda installation/bin/conda"),
            PATH=f"{path_python.parent}{os.pathsep}{self.env['PATH']}",
        )

    def test_incomplete_active_conda_falls_back_to_checkout_venv(self):
        active = self.interpreter("active conda", missing_numpy=True)
        local = self.interpreter(".venv")
        self.assert_selected(local, CONDA_PREFIX=str(active.parent.parent))

    def test_incompatible_lerobot_falls_back_to_checkout_venv(self):
        active = self.interpreter("active conda", incompatible_lerobot=True)
        local = self.interpreter(".venv")
        self.assert_selected(local, CONDA_PREFIX=str(active.parent.parent))

    def test_dry_run_probes_dependencies_without_running_norms_or_training(self):
        self.assert_selected(self.interpreter(".venv"))
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertEqual([call["args"] for call in calls], [["-"]])

    def test_missing_numpy_stops_before_norms_and_reports_the_interpreter(self):
        active = self.interpreter("explicit", missing_numpy=True)
        self.interpreter(".venv")
        result = self.launch(PYTHON=str(active), DRY_RUN="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Requested Python is not a usable training environment", result.stderr)
        self.assertIn("numpy", result.stderr)
        self.assertIn(str(active), result.stderr)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertEqual([call["args"] for call in calls], [["-"]])

    def test_same_active_python_runs_checks_norms_and_training(self):
        self.interpreter(".venv")
        active = self.interpreter("active conda")
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
