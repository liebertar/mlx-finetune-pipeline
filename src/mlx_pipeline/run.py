"""CLI entry point: python -m mlx_pipeline.run --config configs/llm_lora.yaml"""

import argparse
import os
from pathlib import Path

from mlx_pipeline.config import load_config
from mlx_pipeline.pipelines import finetune_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run an MLX fine-tuning pipeline")
    parser.add_argument("--config", required=True, help="path to a YAML in configs/")
    parser.add_argument("--work-dir", default="outputs", help="where adapters/models are written")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
    cfg = load_config(args.config)
    work_dir = Path(args.work_dir) / Path(args.config).stem
    finetune_pipeline(cfg=cfg, work_dir=str(work_dir))


if __name__ == "__main__":
    main()
