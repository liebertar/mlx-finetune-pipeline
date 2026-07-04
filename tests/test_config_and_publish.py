import pytest

from mlx_pipeline.config import load_config
from mlx_pipeline.steps.publish import render_model_card
from mlx_pipeline.steps.train import build_command


@pytest.mark.parametrize(
    "path", ["configs/llm_lora.yaml", "configs/vlm_lora.yaml", "configs/embedding_sft.yaml"]
)
def test_bundled_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg.model
    assert cfg.publish["repo_id"]


def test_config_rejects_unknown_flavor(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("flavor: audio\nmodel: x\n")
    with pytest.raises(ValueError, match="flavor"):
        load_config(bad)


def test_train_command_dispatches_on_flavor():
    llm = build_command("llm", "m", "d", {"method": "lora"}, "out")
    vlm = build_command("vlm", "m", "d", {}, "out")
    assert "mlx_lm" in llm
    assert "mlx_vlm.lora" in vlm


def test_model_card_includes_metrics():
    card = render_model_card("org/model", "base/model", {"eval_loss": 1.2345})
    assert "org/model" in card
    assert "1.2345" in card
    assert "base/model" in card
