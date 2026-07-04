"""Hyperparameter sweep driver.

The search space lives in the config's `sweep:` section — combos (any train.*
keys incl. batch_size), the selection metric, and the epoch budget. Each combo
trains the SAME base model on the SAME data (sequentially — MLX saturates the
GPU) and is scored on the SAME fixed gold split. iters are derived from
`epochs` per combo so different batch sizes get an equal number of data passes.

Results: sweep.json + sweep_comparison.png land in the main work dir, so the
next publish run carries the comparison into the model card.

Usage:  python scripts/sweep.py --config configs/pii_masking.yaml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlx_pipeline.charts import plot_sweep_comparison  # noqa: E402
from mlx_pipeline.config import load_config  # noqa: E402

# fallback space for configs without a sweep: section
DEFAULT_SWEEP = {
    "select": "final_val_loss",
    "higher_is_better": False,
    "combos": [
        {"label": "r8-lr1e-5", "rank": 8, "learning_rate": 1e-5},
        {"label": "r16-lr1e-4", "rank": 16, "learning_rate": 1e-4},
        {"label": "r32-lr2e-4", "rank": 32, "learning_rate": 2e-4},
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm_lora.yaml")
    args = parser.parse_args()
    stem = Path(args.config).stem
    cfg = load_config(args.config)
    sweep_cfg = cfg.sweep or DEFAULT_SWEEP
    combos = [dict(c) for c in sweep_cfg["combos"]]
    select = sweep_cfg.get("select", "final_val_loss")
    higher = sweep_cfg.get("higher_is_better", select != "final_val_loss")
    epochs = sweep_cfg.get("epochs")

    n_data = len(Path(cfg.data["source"]).read_text().splitlines())
    n_train = round(n_data * (1 - cfg.data.get("valid_split", 0.1)
                              - cfg.data.get("test_split", 0.1)))

    for combo in combos:
        work_dir = f"outputs/sweep/{stem}/{combo['label']}"
        overrides = []
        for key, value in combo.items():
            if key == "label":
                continue
            overrides += ["--set", f"train.{key}={value}"]
        if epochs:
            batch = combo.get("batch_size", cfg.train.get("batch_size", 2))
            combo["iters"] = max(1, round(n_train * epochs / batch))
            overrides += ["--set", f"train.iters={combo['iters']}"]
        for key, value in sweep_cfg.get("eval_overrides", {}).items():
            overrides += ["--set", f"eval.{key}={value}"]

        print(f"=== sweep: {combo['label']} ({overrides}) ===", flush=True)
        result = subprocess.run([
            sys.executable, "-m", "mlx_pipeline.run",
            "--config", args.config, "--work-dir", work_dir, "--skip-publish",
            *overrides,
        ])
        combo["gate_passed"] = result.returncode == 0

        metrics_file = Path(work_dir) / "metrics.json"
        if metrics_file.exists():
            combo.update(json.loads(metrics_file.read_text()))
        history_file = Path(work_dir) / "history.json"
        combo["val"] = (json.loads(history_file.read_text())["val"]
                        if history_file.exists() else [])
        combo["final_val_loss"] = combo["val"][-1][1] if combo["val"] else None

    scored = [c for c in combos if c.get(select) is not None]
    if not scored:
        sys.exit("sweep produced no results")
    best = (max if higher else min)(scored, key=lambda c: c[select])
    best["selected"] = True

    out = Path("outputs") / stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(
        [{k: v for k, v in c.items() if k != "val"} for c in combos],
        ensure_ascii=False, indent=2,
    ))
    plot_sweep_comparison(combos, str(out / "sweep_comparison.png"))

    print(json.dumps({c["label"]: c.get(select) for c in combos}, indent=2))
    winner_sets = " ".join(
        f"--set train.{k}={best[k]}" for k in best
        if k in ("rank", "learning_rate", "batch_size", "iters", "scale")
    )
    print(f"BEST: {best['label']} ({select}={best[select]})")
    print(f"publish with:  python -m mlx_pipeline.run --config {args.config} {winner_sets}")


if __name__ == "__main__":
    main()
