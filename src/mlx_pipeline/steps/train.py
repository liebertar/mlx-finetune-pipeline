"""Node 3 — train: run MLX fine-tuning on the host GPU (Metal).

Dispatches on config flavor:
  llm / embedding -> mlx_lm.lora
  vlm             -> mlx_vlm.lora

Runs the trainers as subprocesses (their supported entry points) and logs
params to MLflow. Requires native macOS — this cannot run inside Docker.
"""

import subprocess
import sys
from pathlib import Path

import mlflow
from zenml import step


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
    if flavor in ("llm", "embedding") and train_cfg.get("method") in ("lora", "dora"):
        cmd += ["--fine-tune-type", train_cfg["method"]]
    return cmd


@step
def train(flavor: str, model: str, dataset_dir: str, train_cfg: dict, work_dir: str) -> str:
    cmd = build_command(flavor, model, dataset_dir, train_cfg, work_dir)
    mlflow.log_params({"flavor": flavor, "model": model, **train_cfg})
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"training failed:\n{result.stderr[-2000:]}")
    adapter_path = Path(work_dir) / "adapters"
    mlflow.log_artifacts(str(adapter_path), artifact_path="adapters")
    return str(adapter_path)
