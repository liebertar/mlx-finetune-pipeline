"""Node 3 — train: run MLX fine-tuning on the host GPU (Metal).

Dispatches on config flavor:
  llm / embedding -> mlx_lm.lora
  vlm             -> mlx_vlm.lora

Every research-backed knob from docs/TUNING.md is wired through:
rank/scale/dropout and the cosine-warmup LR schedule go via a generated
`-c` YAML (mlx-lm has no CLI flags for them); mask_prompt, seed,
grad-accumulation, grad-checkpoint, max_seq_length, and num_layers go via
CLI flags. Per-iteration train/val losses are parsed from trainer output,
logged to MLflow as step metrics, and rendered to a loss-curve PNG.
Requires native macOS — this cannot run inside Docker.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import mlflow
import yaml
from zenml import step

from mlx_pipeline.charts import plot_loss_curve

TRAIN_LINE = re.compile(r"Iter (\d+): Train loss ([\d.]+)")
VAL_LINE = re.compile(r"Iter (\d+): Val loss ([\d.]+)")

# train.* keys that map 1:1 to mlx_lm CLI flags
FLAG_KEYS = {
    "grad_accumulation_steps": "--grad-accumulation-steps",
    "max_seq_length": "--max-seq-length",
    "num_layers": "--num-layers",
    "seed": "--seed",
}


def _mlx_lm_config(train_cfg: dict, out_dir: Path) -> Path | None:
    """rank/scale/dropout and lr_schedule have no CLI flags — write the -c YAML."""
    cfg: dict = {}
    if "rank" in train_cfg:
        cfg["lora_parameters"] = {
            "rank": train_cfg["rank"],
            "scale": train_cfg.get("scale", 20.0),
            "dropout": train_cfg.get("dropout", 0.0),
        }
    if train_cfg.get("lr_schedule") == "cosine":
        iters = train_cfg.get("iters", 200)
        lr = train_cfg.get("learning_rate", 1e-5)
        warmup = max(1, int(iters * train_cfg.get("warmup_fraction", 0.1)))
        cfg["lr_schedule"] = {
            "name": "cosine_decay",
            "warmup": warmup,
            "warmup_init": 0.0,
            "arguments": [lr, iters - warmup, lr * 0.01],
        }
    if not cfg:
        return None
    path = out_dir / "mlx_lm_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg))
    return path


def build_command(flavor: str, model: str, dataset_dir: str, train_cfg: dict, out_dir: str) -> list[str]:
    if flavor in ("llm", "embedding"):
        cmd = [sys.executable, "-m", "mlx_lm", "lora", "--train"]
    elif flavor == "vlm":
        cmd = [sys.executable, "-m", "mlx_vlm.lora"]
    else:
        raise ValueError(f"unknown flavor: {flavor}")
    cmd += [
        "--model", model,
        "--data", dataset_dir,
        "--adapter-path", str(Path(out_dir) / "adapters"),
        "--iters", str(train_cfg.get("iters", 200)),
        "--batch-size", str(train_cfg.get("batch_size", 2)),
        "--learning-rate", str(train_cfg.get("learning_rate", 1e-5)),
    ]
    if flavor in ("llm", "embedding"):
        if train_cfg.get("method") in ("lora", "dora"):
            cmd += ["--fine-tune-type", train_cfg["method"]]
        # dense report/eval intervals so the loss curve has real resolution
        cmd += [
            "--steps-per-report", str(train_cfg.get("steps_per_report", 10)),
            "--steps-per-eval", str(train_cfg.get("steps_per_eval", 25)),
        ]
        for key, flag in FLAG_KEYS.items():
            if key in train_cfg:
                cmd += [flag, str(train_cfg[key])]
        if train_cfg.get("mask_prompt"):
            cmd += ["--mask-prompt"]
        if train_cfg.get("grad_checkpoint"):
            cmd += ["--grad-checkpoint"]
        if cfg_path := _mlx_lm_config(train_cfg, Path(out_dir)):
            cmd += ["-c", str(cfg_path)]
    return cmd


def parse_history(output: str) -> dict:
    """Extract per-iteration losses from mlx_lm trainer output."""
    return {
        "train": [[int(i), float(l)] for i, l in TRAIN_LINE.findall(output)],
        "val": [[int(i), float(l)] for i, l in VAL_LINE.findall(output)],
    }


@step
def train(flavor: str, model: str, dataset_dir: str, train_cfg: dict, work_dir: str) -> str:
    cmd = build_command(flavor, model, dataset_dir, train_cfg, work_dir)
    mlflow.log_params({"flavor": flavor, "model": model, **train_cfg})
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"training failed:\n{result.stderr[-2000:]}")

    history = parse_history(result.stdout + result.stderr)
    for it, loss in history["train"]:
        mlflow.log_metric("train_loss", loss, step=it)
    for it, loss in history["val"]:
        mlflow.log_metric("val_loss", loss, step=it)

    work = Path(work_dir)
    (work / "history.json").write_text(json.dumps(history))
    plot_loss_curve(history, str(work / "loss_curve.png"))
    mlflow.log_artifact(str(work / "loss_curve.png"))

    adapter_path = work / "adapters"
    mlflow.log_artifacts(str(adapter_path), artifact_path="adapters")
    return str(adapter_path)
