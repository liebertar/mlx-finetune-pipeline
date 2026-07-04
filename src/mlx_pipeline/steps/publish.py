"""Node 7 — publish: upload the fused model plus an auto-generated model card
to the Hugging Face Hub. Needs HF_TOKEN in the environment."""

import os
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
[mlx-finetune-pipeline](https://github.com/your-org/mlx-finetune-pipeline)
(MLX on Apple Silicon, ZenML + MLflow).

## Eval metrics

{metrics_table}
"""


def render_model_card(repo_id: str, base_model: str, metrics: dict) -> str:
    rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items())
    table = "| metric | value |\n|---|---|\n" + rows if metrics else "_none recorded_"
    return CARD_TEMPLATE.format(repo_id=repo_id, base_model=base_model, metrics_table=table)


@step
def publish(model_dir: str, base_model: str, metrics: dict, publish_cfg: dict) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set — the publish node cannot upload")
    repo_id = publish_cfg["repo_id"]
    card = render_model_card(repo_id, base_model, metrics)
    Path(model_dir, "README.md").write_text(card)

    api = HfApi(token=token)
    api.create_repo(repo_id, private=publish_cfg.get("private", True), exist_ok=True)
    api.upload_folder(folder_path=model_dir, repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"
