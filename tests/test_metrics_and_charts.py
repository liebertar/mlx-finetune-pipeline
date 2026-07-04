from mlx_pipeline.charts import plot_loss_curve, plot_sweep_comparison
from mlx_pipeline.steps.publish import render_model_card
from mlx_pipeline.steps.train import build_command, parse_history

TRAINER_OUTPUT = """
Loading pretrained model
Iter 1: Val loss 3.291, Val took 1.2s
Iter 10: Train loss 2.585, Learning Rate 1.000e-05, It/sec 3.5
Iter 20: Train loss 1.902, Learning Rate 1.000e-05, It/sec 3.6
Iter 25: Val loss 1.700, Val took 1.1s
Iter 30: Train loss 1.204, Learning Rate 1.000e-05, It/sec 3.6
Saved final weights to adapters.safetensors.
"""


def test_parse_history_extracts_both_series():
    h = parse_history(TRAINER_OUTPUT)
    assert h["train"] == [[10, 2.585], [20, 1.902], [30, 1.204]]
    assert h["val"] == [[1, 3.291], [25, 1.7]]


def test_parse_history_empty_output():
    h = parse_history("no metrics here")
    assert h == {"train": [], "val": []}


def test_build_command_writes_rank_config(tmp_path):
    cmd = build_command("llm", "m", "d", {"method": "lora", "rank": 16}, str(tmp_path))
    assert "-c" in cmd
    cfg = (tmp_path / "mlx_lm_config.yaml").read_text()
    assert "rank: 16" in cfg


def test_loss_curve_png(tmp_path):
    h = parse_history(TRAINER_OUTPUT)
    out = plot_loss_curve(h, str(tmp_path / "loss.png"))
    assert (tmp_path / "loss.png").stat().st_size > 1000


def test_sweep_comparison_png(tmp_path):
    runs = [
        {"label": "r8 lr1e-5", "val": [[1, 3.0], [25, 1.5]]},
        {"label": "r16 lr1e-5", "val": [[1, 3.1], [25, 1.2]]},
    ]
    plot_sweep_comparison(runs, str(tmp_path / "sweep.png"))
    assert (tmp_path / "sweep.png").stat().st_size > 1000


def test_model_card_with_sweep_and_curve():
    sweep = [
        {"label": "r8 lr1e-5", "rank": 8, "learning_rate": 1e-5,
         "final_val_loss": 0.09, "selected": True},
        {"label": "r8 lr5e-5", "rank": 8, "learning_rate": 5e-5,
         "final_val_loss": 0.21},
    ]
    card = render_model_card(
        "org/model", "base/model", {"eval_loss": 0.05},
        train_cfg={"rank": 8, "learning_rate": 1e-5},
        has_loss_curve=True, sweep=sweep,
    )
    assert "loss_curve.png" in card
    assert "sweep_comparison.png" in card
    assert "**selected**" in card
    assert "| rank | 8 |" in card
