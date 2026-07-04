"""Node 4 — evaluate: score the tuned adapter on the held-out test split.

Produces eval_loss and eval_ppl (perplexity = exp(loss)), plus — for the llm
flavor — sample generations from the test prompts so a human can judge
whether the tuning matches the *intent*, not just the loss. Everything is
logged to MLflow."""

import json
import re
import subprocess
import sys
from math import exp
from pathlib import Path

import mlflow
from zenml import step


def parse_test_loss(output: str) -> float:
    match = re.search(r"Test loss ([\d.]+)", output)
    if not match:
        raise RuntimeError(f"could not find test loss in trainer output:\n{output[-2000:]}")
    return float(match.group(1))


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"evaluation failed:\n{result.stderr[-2000:]}")
    return result.stdout + result.stderr


def sample_generations(
    model: str, adapter_path: str, dataset_dir: str, n: int = 3, max_tokens: int = 120
) -> list[dict]:
    """Generate completions for the first n test prompts with the tuned adapter."""
    samples = []
    test_file = Path(dataset_dir) / "test.jsonl"
    for line in test_file.read_text().splitlines()[:n]:
        rec = json.loads(line)
        prompt = next(
            (m["content"] for m in rec.get("messages", []) if m["role"] == "user"), None
        )
        if prompt is None:
            continue
        out = _run([
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", model,
            "--adapter-path", adapter_path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
        ])
        # mlx_lm generate prints the completion between ========== markers
        parts = out.split("==========")
        completion = parts[1].strip() if len(parts) >= 2 else out.strip()
        reference = next(
            (m["content"] for m in rec["messages"] if m["role"] == "assistant"), ""
        )
        samples.append({"prompt": prompt, "generated": completion, "reference": reference})
    return samples


@step
def evaluate(flavor: str, model: str, dataset_dir: str, adapter_path: str, work_dir: str) -> dict:
    if flavor in ("llm", "embedding"):
        output = _run([
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", model,
            "--data", dataset_dir,
            "--adapter-path", adapter_path,
            "--test",
            # tiny test splits (smoke datasets) are smaller than the default
            # eval batch size of 4
            "--batch-size", "1",
        ])
        loss = parse_test_loss(output)
        metrics = {"eval_loss": loss, "eval_ppl": exp(loss)}
    else:
        # mlx-vlm has no --test mode yet; TODO: sample generations + scoring
        raise NotImplementedError("vlm evaluation not implemented yet")

    if flavor == "llm":
        samples = sample_generations(model, adapter_path, dataset_dir)
        report = "\n\n---\n\n".join(
            f"**prompt:** {s['prompt']}\n\n**generated:** {s['generated']}\n\n"
            f"**reference:** {s['reference']}"
            for s in samples
        )
        sample_file = Path(work_dir) / "sample_generations.md"
        sample_file.write_text(report or "_no samples_")
        mlflow.log_artifact(str(sample_file))

    mlflow.log_metrics(metrics)
    return metrics
