"""Hyperparameter sweep driver: trains the SAME base model over several
LoRA-rank / learning-rate combinations (sequentially — MLX saturates the GPU,
parallel runs would thrash unified memory), compares final validation loss,
and drops sweep.json + sweep_comparison.png into the main work dir so the next
publish run carries the comparison into the model card.

Usage:  python scripts/sweep.py [--config configs/llm_lora.yaml]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlx_pipeline.charts import plot_sweep_comparison  # noqa: E402

# The three recipes from docs/TUNING.md: mlx-lm's timid default as baseline,
# the consensus SFT recipe, and the aggressive end of the recommended range.
COMBOS = [
    {"label": "r8-lr1e-5", "rank": 8, "learning_rate": 1e-5},
    {"label": "r16-lr1e-4", "rank": 16, "learning_rate": 1e-4},
    {"label": "r32-lr2e-4", "rank": 32, "learning_rate": 2e-4},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm_lora.yaml")
    args = parser.parse_args()
    stem = Path(args.config).stem

    for combo in COMBOS:
        work_dir = f"outputs/sweep/{stem}/{combo['label']}"
        print(f"=== sweep: {combo['label']} ===", flush=True)
        result = subprocess.run([
            sys.executable, "-m", "mlx_pipeline.run",
            "--config", args.config,
            "--work-dir", work_dir,
            "--skip-publish",
            "--set", f"train.rank={combo['rank']}",
            "--set", f"train.learning_rate={combo['learning_rate']}",
        ])
        combo["gate_passed"] = result.returncode == 0
        history_file = Path(work_dir) / "history.json"
        if history_file.exists():
            combo["val"] = json.loads(history_file.read_text())["val"]
        else:
            combo["val"] = []
        combo["final_val_loss"] = combo["val"][-1][1] if combo["val"] else float("inf")

    scored = [c for c in COMBOS if c["val"]]
    if not scored:
        sys.exit("sweep produced no results")
    best = min(scored, key=lambda c: c["final_val_loss"])
    best["selected"] = True

    out = Path("outputs") / stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(
        [{k: v for k, v in c.items() if k != "val"} for c in COMBOS], indent=2
    ))
    plot_sweep_comparison(COMBOS, str(out / "sweep_comparison.png"))

    print(json.dumps({c["label"]: c["final_val_loss"] for c in COMBOS}, indent=2))
    print(f"BEST: {best['label']} (rank={best['rank']}, lr={best['learning_rate']}, "
          f"val_loss={best['final_val_loss']:.4f})")
    print(f"publish with:  python -m mlx_pipeline.run --config {args.config} "
          f"--set train.rank={best['rank']} --set train.learning_rate={best['learning_rate']}")


if __name__ == "__main__":
    main()
