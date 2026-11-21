"""Launch from the repository root: uv run modal run scripts/modal_sft.py.

Defaults to a three-step FP8 H100 smoke test. Pass --full-run for 300 steps.
Only scripts/ and the prepared GSM8K split (if present) are uploaded, never
credentials or the local Python environment.
"""

from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
STORE = "/store"
app = modal.App("qwen3-gsm8k-sft")
# Container disks are ephemeral; the Volume holds reusable data and checkpoints.
volume = modal.Volume.from_name("qwen3-gsm8k", create_if_missing=True)

# Share dependencies with local development; platform markers select CUDA wheels.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"HF_HOME": f"{STORE}/hf", "PYTHONPATH": "/project", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "scripts", remote_path="/project/scripts",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
)
if (ROOT / "data/gsm8k/train").is_dir():
    image = image.add_local_dir(ROOT / "data/gsm8k", remote_path="/seed-data")


@app.function(image=image, volumes={STORE: volume}, timeout=1800, cpu=2, memory=8192,
              scaledown_window=2)
def prepare():
    """Download/cache on CPU so setup doesn't consume H100 time."""
    import shutil
    import subprocess
    import sys

    from huggingface_hub import snapshot_download
    from scripts.data import MODEL_ID

    destination = Path(STORE) / "data/gsm8k"
    if not (destination / "train").exists():
        if Path("/seed-data").exists():
            shutil.copytree("/seed-data", destination, dirs_exist_ok=True)
        else:
            subprocess.run(
                [sys.executable, "-m", "scripts.prepare_data", "--output", str(destination)],
                cwd="/project", check=True,
            )
    snapshot_download(MODEL_ID)
    volume.commit()


@app.function(
    image=image, gpu="H100", volumes={STORE: volume},
    timeout=3600, cpu=2, memory=16384, retries=0, scaledown_window=2,
)
def train(run_name: str, smoke_test: bool, steps: int, batch_size: int,
          gradient_accumulation_steps: int, gradient_checkpointing: bool,
          eval_batch_size: int, max_length: int, eval_steps: int,
          logging_steps: int, save_steps: int, eval_on_start: bool):
    """Run CUDA training remotely and commit artifacts even when the subprocess fails."""
    import subprocess
    import sys

    # Refresh this container's view after the CPU preparation job commits its files.
    volume.reload()
    output = f"{STORE}/outputs/{run_name}"
    if Path(output).exists():
        raise RuntimeError(f"Output already exists: {output}; choose a new run name")
    command = [
        sys.executable, "-m", "scripts.train_sft",
        "--data", f"{STORE}/data/gsm8k", "--output", output,
        "--precision", "fp8", "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--eval-batch-size", str(eval_batch_size), "--max-length", str(max_length),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--eval-steps", str(eval_steps), "--logging-steps", str(logging_steps),
        "--save-steps", str(save_steps),
        "--eval-on-start" if eval_on_start else "--no-eval-on-start",
    ]
    if gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if smoke_test:
        command.append("--smoke-test")
    try:
        subprocess.run(command, cwd="/project", check=True)
        if smoke_test:
            # A fresh process loads the saved adapter onto an ordinary base model.
            subprocess.run([
                sys.executable, "-m", "scripts.evaluate",
                "--data", f"{STORE}/data/gsm8k", "--adapter", output,
                "--split", "validation", "--max-samples", "3",
                "--max-new-tokens", "128", "--out", f"{output}/reload_eval.json",
            ], cwd="/project", check=True)
    finally:
        volume.commit()
    return output


@app.local_entrypoint()
def main(full_run: bool = False, steps: int = 300, run_name: str = "",
         batch_size: int = 16, gradient_accumulation_steps: int = 1,
         gradient_checkpointing: bool = False, eval_batch_size: int = 16,
         max_length: int = 512, eval_steps: int = 25, logging_steps: int = 5,
         save_steps: int = 100, eval_on_start: bool = True):
    import re
    from datetime import datetime, timezone
    from uuid import uuid4

    if steps < 1:
        raise ValueError("steps must be positive")
    if min(eval_steps, logging_steps, save_steps) < 1:
        raise ValueError("eval-steps, logging-steps and save-steps must be positive")
    if min(batch_size, gradient_accumulation_steps, eval_batch_size, max_length) < 1:
        raise ValueError("batch sizes, max-length and accumulation must be positive")
    if not run_name:
        label = "sft" if full_run else "smoke"
        run_name = f"{label}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run-name must contain only letters, digits, underscores and hyphens")
    print(f"Run {run_name}: {'full training' if full_run else '3-step smoke test'} on one H100")
    # Finish downloads on CPU before allocating the training GPU.
    prepare.remote()
    output = train.remote(run_name, not full_run, steps, batch_size,
                          gradient_accumulation_steps, gradient_checkpointing,
                          eval_batch_size, max_length, eval_steps,
                          logging_steps, save_steps, eval_on_start)
    print(f"Saved to Modal volume qwen3-gsm8k: {output.removeprefix(STORE + '/')}")
    print(f"Download: uv run modal volume get qwen3-gsm8k outputs/{run_name} ./outputs/{run_name}")
