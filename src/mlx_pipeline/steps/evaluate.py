"""Node 4 — evaluate: score the tuned adapter on the held-out test split
and log metrics to MLflow."""

import re
import subprocess
import sys

import mlflow
from zenml import step


def parse_test_loss(output: str) -> float:
    match = re.search(r"Test loss ([\d.]+)", output)
    if not match:
        raise RuntimeError(f"could not find test loss in trainer output:\n{output[-2000:]}")
    return float(match.group(1))


@step
def evaluate(flavor: str, model: str, dataset_dir: str, adapter_path: str) -> dict:
    if flavor in ("llm", "embedding"):
        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", model,
            "--data", dataset_dir,
            "--adapter-path", adapter_path,
            "--test",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"evaluation failed:\n{result.stderr[-2000:]}")
        metrics = {"eval_loss": parse_test_loss(result.stdout + result.stderr)}
    else:
        # mlx-vlm has no --test mode yet; TODO: sample generations + scoring
        raise NotImplementedError("vlm evaluation not implemented yet")
    mlflow.log_metrics(metrics)
    return metrics
