"""Evaluate base and SFT on the same H100, question set, and decoding settings.

Reuse the training image and Volume, but launch a separate evaluation-only app.
Results are written to unique directories so earlier comparisons stay intact.
"""

import modal
from scripts.modal_sft import image, volume, STORE

app = modal.App("qwen3-gsm8k-evaluation")


@app.function(image=image, gpu="H100", volumes={STORE: volume},
              cpu=2, memory=16384, timeout=3600, retries=0, scaledown_window=2)
def evaluate(run_name: str, expected_hash: str, output_name: str):
    """Load each model in a fresh process so weights and GPU allocations cannot leak."""
    import subprocess
    import sys
    from pathlib import Path

    # The adapter was saved by a different job; refresh the mounted Volume first.
    volume.reload()
    adapter = f"{STORE}/outputs/{run_name}"
    if not Path(adapter, "adapter_config.json").exists():
        raise RuntimeError(f"Missing adapter: {adapter}")
    destination = f"{STORE}/results/{output_name}"
    if Path(destination).exists():
        raise RuntimeError("Evaluation output already exists")
    try:
        for label in ("base", "sft"):
            command = [sys.executable, "-m", "scripts.evaluate",
                       "--data", f"{STORE}/data/gsm8k", "--split", "validation",
                       "--max-samples", "0", "--max-new-tokens", "512",
                       "--batch-size", "64", "--expected-dataset-sha256", expected_hash,
                       "--out", f"{destination}/{label}.json"]
            if label == "sft":
                command += ["--adapter", adapter]
            print(f"Evaluating {label}: all validation questions, greedy, 512 tokens, batch 64", flush=True)
            subprocess.run(command, cwd="/project", check=True)
            volume.commit()  # Keep the completed base result if SFT evaluation fails.
    finally:
        volume.commit()
    return destination.removeprefix(STORE + "/")


@app.local_entrypoint()
def main(run_name: str = "sft-20261121-002201-0e7388"):
    import json
    import re
    from pathlib import Path
    from uuid import uuid4

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Invalid run name")
    # The baseline hash pins question content and order, not just the split name.
    baseline = json.loads(Path("results/base_val_full_512.json").read_text())
    destination = evaluate.remote(run_name, baseline["dataset_sha256"], f"{run_name}-{uuid4().hex[:6]}")
    print(f"Download: uv run modal volume get qwen3-gsm8k {destination} ./results/{destination.split('/')[-1]}")
