# adapted from openpi
import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


@pytest.mark.parametrize(
    ("probability", "draw", "dropped"),
    [(0.0, None, False), (1.0, None, True), (0.2, 0.1, True), (0.2, 0.8, False)],
)
def test_state_input_dropout_omits_tokens_and_preserves_state_and_labels(monkeypatch, probability, draw, dropped):
    def sample():
        assert draw is not None, "Boundary probabilities should not consume randomness"
        return draw

    monkeypatch.setattr(np.random, "random", sample)
    encoded_text = []

    class Encoder:
        def encode(self, text, **kwargs):
            encoded_text.append(text)
            return [1, 2]

    # Exercise the real prompt formatting without downloading tokenizer assets.
    tokenizer = object.__new__(_tokenizer.PaligemmaTokenizer)
    tokenizer._max_len = 32  # noqa: SLF001
    tokenizer._tokenizer = Encoder()  # noqa: SLF001
    state = np.arange(8, dtype=np.float32) / 10
    actions = np.ones((30, 8), dtype=np.float32)
    image = {"base_0_rgb": np.zeros((8, 12, 3), dtype=np.uint8)}
    data = {"prompt": "pick up the object", "state": state, "actions": actions, "image": image}
    transform = _transforms.compose(
        [_transforms.StateInputDropout(probability), _transforms.TokenizePrompt(tokenizer, discrete_state_input=True)]
    )
    result = transform(data)
    assert ("State:" not in "".join(encoded_text)) == dropped
    assert "pick up the object" in "".join(encoded_text)
    assert "_drop_state_input" not in result
    assert result["state"] is state
    assert result["actions"] is actions
    assert result["image"] is image
    assert "_drop_state_input" not in data
    assert data["prompt"] == "pick up the object"


def test_state_input_dropout_resamples_for_each_sample(monkeypatch):
    draws = iter([0.1, 0.8])
    monkeypatch.setattr(np.random, "random", lambda: next(draws))
    transform = _transforms.StateInputDropout(0.2)
    assert transform({})["_drop_state_input"]
    assert not transform({})["_drop_state_input"]


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_state_input_dropout_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError, match="probability"):
        _transforms.StateInputDropout(probability)


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
