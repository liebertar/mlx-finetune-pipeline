"""CLI entry point: python -m mlx_pipeline.run --config configs/llm_lora.yaml

--set key=value overrides nested config values (e.g. --set train.rank=16).
--skip-publish stops the pipeline after the quality gate (used for sweeps).
"""

import argparse
import os
from pathlib import Path

from mlx_pipeline.config import load_config
from mlx_pipeline.pipelines import finetune_pipeline


def apply_override(cfg, dotted_key: str, value: str):
    keys = dotted_key.split(".")
    target = getattr(cfg, keys[0]) if len(keys) > 1 else cfg
    for k in keys[1:-1]:
        target = target[k]
    # YAML-ish literal coercion for numbers
    for cast in (int, float):
        try:
            value = cast(value)
            break
        except ValueError:
            continue
    if len(keys) == 1:
        setattr(cfg, keys[0], value)
    else:
        target[keys[-1]] = value


def main():
    parser = argparse.ArgumentParser(description="Run an MLX fine-tuning pipeline")
    parser.add_argument("--config", required=True, help="path to a YAML in configs/")
    parser.add_argument("--work-dir", default=None, help="where adapters/models are written")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.rank=16")
    parser.add_argument("--skip-publish", action="store_true",
                        help="stop after the quality gate (no fuse/upload)")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
    cfg = load_config(args.config)
    for override in args.set:
        key, _, value = override.partition("=")
        apply_override(cfg, key, value)

    work_dir = args.work_dir or str(Path("outputs") / Path(args.config).stem)
    finetune_pipeline(cfg=cfg, work_dir=work_dir, publish_enabled=not args.skip_publish)


if __name__ == "__main__":
    main()
