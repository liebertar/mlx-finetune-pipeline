import yaml

from mlx_pipeline.steps.ingest import dedupe
from mlx_pipeline.steps.train import build_command


def test_dedupe_drops_exact_duplicates():
    rec = {"messages": [{"role": "user", "content": "hi"}]}
    other = {"messages": [{"role": "user", "content": "yo"}]}
    unique, dropped = dedupe([rec, dict(rec), other, dict(rec)])
    assert unique == [rec, other]
    assert dropped == 2


def test_dedupe_noop_on_unique():
    records = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(5)]
    unique, dropped = dedupe(records)
    assert len(unique) == 5 and dropped == 0


FULL_CFG = {
    "method": "lora", "rank": 16, "scale": 2.0, "learning_rate": 1e-4,
    "lr_schedule": "cosine", "warmup_fraction": 0.1, "iters": 200,
    "batch_size": 2, "grad_accumulation_steps": 4, "mask_prompt": True,
    "seed": 42, "grad_checkpoint": True, "max_seq_length": 2048,
}


def test_build_command_wires_all_flags(tmp_path):
    cmd = build_command("llm", "m", "d", FULL_CFG, str(tmp_path))
    for flag in ("--mask-prompt", "--grad-checkpoint", "--seed",
                 "--grad-accumulation-steps", "--max-seq-length", "-c"):
        assert flag in cmd, f"missing {flag}"
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert cmd[cmd.index("--grad-accumulation-steps") + 1] == "4"


def test_generated_mlx_config_has_rank_scale_and_schedule(tmp_path):
    build_command("llm", "m", "d", FULL_CFG, str(tmp_path))
    cfg = yaml.safe_load((tmp_path / "mlx_lm_config.yaml").read_text())
    assert cfg["lora_parameters"] == {"rank": 16, "scale": 2.0, "dropout": 0.0}
    sched = cfg["lr_schedule"]
    assert sched["name"] == "cosine_decay"
    assert sched["warmup"] == 20                      # 10% of 200 iters
    assert sched["arguments"][0] == 1e-4              # init lr
    assert sched["arguments"][1] == 180               # decay steps after warmup


def test_no_config_file_when_no_rank_or_schedule(tmp_path):
    cmd = build_command("llm", "m", "d", {"method": "lora"}, str(tmp_path))
    assert "-c" not in cmd
    assert not (tmp_path / "mlx_lm_config.yaml").exists()
