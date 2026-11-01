from types import SimpleNamespace

import pytest

from openpi import transforms
from openpi.policies import policy_config


@pytest.mark.parametrize("use_inference_hook", [False, True])
def test_default_loader_keeps_original_policy_constructor_and_optional_factory_is_generic(
    monkeypatch, tmp_path, use_inference_hook
):
    model = object()
    loaded = []

    def ordinary(value, **kwargs):
        loaded.append(("ordinary", value, kwargs))
        return "ordinary"

    def alternative(value, **kwargs):
        loaded.append(("alternative", value, kwargs))
        return "alternative"

    data = SimpleNamespace(
        asset_id="unused",
        use_quantile_norm=False,
        data_transforms=transforms.Group(),
        model_transforms=transforms.Group(),
    )
    config = SimpleNamespace(
        model=SimpleNamespace(load=lambda weights: model),
        data=SimpleNamespace(create=lambda *args: data),
        assets_dirs=tmp_path,
        policy_metadata={"fps": 30},
    )
    if use_inference_hook:
        def training_only(*args):
            raise AssertionError("Serving must not request training reward labels")

        config.data.create = training_only
        config.data.create_for_inference = lambda *args: data
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda value: tmp_path)
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *a, **k: {})  # noqa: SLF001
    monkeypatch.setattr(policy_config._policy, "Policy", ordinary)  # noqa: SLF001
    assert policy_config.create_trained_policy(config, tmp_path, norm_stats={}) == "ordinary"
    assert (
        policy_config.create_trained_policy(config, tmp_path, norm_stats={}, policy_factory=alternative)
        == "alternative"
    )
    assert [item[0] for item in loaded] == ["ordinary", "alternative"]
    assert loaded[0][1] is loaded[1][1] is model
    assert loaded[0][2] == loaded[1][2]
