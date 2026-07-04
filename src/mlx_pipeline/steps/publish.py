"""Node 7 — publish: upload the fused model plus an auto-generated model card
(hyperparameters, eval metrics, training loss curve, and — when a sweep was
run — the hyperparameter comparison) to the Hugging Face Hub.
Needs HF_TOKEN in the environment."""

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi
from zenml import step

CARD_TEMPLATE = """---
library_name: mlx
base_model: {base_model}
tags:
  - mlx
  - fine-tuned
  - mlx-finetune-pipeline
---

# {repo_id}

Fine-tuned from `{base_model}` with the
[mlx-finetune-pipeline](https://github.com/liebertar/mlx-finetune-pipeline)
(MLX on Apple Silicon, ZenML + MLflow).

## Training configuration

{params_table}

## Eval metrics

{metrics_table}
{loss_curve_section}
{sweep_section}
{generations_section}"""

LOSS_CURVE_SECTION = """
## Training loss

![Training loss curve](loss_curve.png)
"""

SWEEP_SECTION = """
## Hyperparameter sweep

This configuration was selected from a sweep over {n} combinations
(same base model, same data, same iterations):

{sweep_table}

![Validation loss by hyperparameter combination](sweep_comparison.png)
"""


def _md_table(header: tuple, rows: list[tuple]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_model_card(
    repo_id: str,
    base_model: str,
    metrics: dict,
    train_cfg: dict | None = None,
    has_loss_curve: bool = False,
    sweep: list[dict] | None = None,
    generations_md: str | None = None,
) -> str:
    metrics_table = (
        _md_table(("metric", "value"), [(k, f"{v:.4f}") for k, v in metrics.items()])
        if metrics else "_none recorded_"
    )
    params_table = (
        _md_table(("parameter", "value"), sorted(train_cfg.items()))
        if train_cfg else "_none recorded_"
    )
    sweep_section = ""
    if sweep:
        rows = [
            (r["label"], r["rank"], r["learning_rate"], f"{r['final_val_loss']:.4f}",
             "**selected**" if r.get("selected") else "")
            for r in sweep
        ]
        sweep_table = _md_table(
            ("combination", "LoRA rank", "learning rate", "final val loss", ""), rows
        )
        sweep_section = SWEEP_SECTION.format(n=len(sweep), sweep_table=sweep_table)
    generations_section = (
        f"\n## Sample generations (held-out prompts)\n\n{generations_md}\n"
        if generations_md else ""
    )
    return CARD_TEMPLATE.format(
        repo_id=repo_id,
        base_model=base_model,
        params_table=params_table,
        metrics_table=metrics_table,
        loss_curve_section=LOSS_CURVE_SECTION if has_loss_curve else "",
        sweep_section=sweep_section,
        generations_section=generations_section,
    )


# no cache: uploading is a side effect, must happen every run
@step(enable_cache=False)
def publish(
    model_dir: str, base_model: str, metrics: dict, publish_cfg: dict,
    train_cfg: dict, work_dir: str,
) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set — the publish node cannot upload")
    repo_id = publish_cfg["repo_id"]
    work = Path(work_dir)

    # bring the charts produced by earlier nodes (and an optional sweep) along
    has_loss_curve = (work / "loss_curve.png").exists()
    if has_loss_curve:
        shutil.copy(work / "loss_curve.png", Path(model_dir) / "loss_curve.png")
    sweep = None
    if (work / "sweep.json").exists():
        sweep = json.loads((work / "sweep.json").read_text())
        if (work / "sweep_comparison.png").exists():
            shutil.copy(work / "sweep_comparison.png", Path(model_dir) / "sweep_comparison.png")
    generations_md = None
    if (work / "sample_generations.md").exists():
        generations_md = (work / "sample_generations.md").read_text()

    card = render_model_card(
        repo_id, base_model, metrics,
        train_cfg=train_cfg, has_loss_curve=has_loss_curve, sweep=sweep,
        generations_md=generations_md,
    )
    Path(model_dir, "README.md").write_text(card)

    api = HfApi(token=token)
    api.create_repo(repo_id, private=publish_cfg.get("private", True), exist_ok=True)
    api.upload_folder(folder_path=model_dir, repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"
