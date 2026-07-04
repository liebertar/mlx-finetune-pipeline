"""Node 6 — fuse: merge the LoRA adapter into the base weights, producing a
standalone MLX model directory ready for distribution."""

import subprocess
import sys
from pathlib import Path

from zenml import step


@step
def fuse(flavor: str, model: str, adapter_path: str, work_dir: str) -> str:
    out_dir = Path(work_dir) / "fused_model"
    module = "mlx_lm" if flavor in ("llm", "embedding") else "mlx_vlm"
    cmd = [
        sys.executable, "-m", module, "fuse",
        "--model", model,
        "--adapter-path", adapter_path,
        "--save-path", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"fuse failed:\n{result.stderr[-2000:]}")
    return str(out_dir)
