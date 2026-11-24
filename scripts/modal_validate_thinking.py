"""Run validation-only greedy and sampling comparisons on saved SFT adapters."""
from pathlib import Path

import modal

from scripts.modal_sft import STORE, image, volume

app = modal.App("qwen3-thinking-sft-validation")
BASELINE = "sft-20261121-002201-0e7388"
CONTINUATION = "sft-think-fp8-continue-bf16-20261123-224223-resume"
CONTROL_HASH = "fefd64ed3f1dbf539f3e1d2a7c4a53bce3413eed7ade72099c6a8f96e0458409"


@app.function(image=image, gpu="H100", volumes={STORE: volume}, cpu=3, memory=16384,
              timeout=5400, retries=0, scaledown_window=2)
def evaluate(run_name: str, label: str, adapter_name: str, sample_shard: int = -1):
    import json
    import subprocess
    import sys

    volume.reload()
    destination = Path(STORE, "results", run_name, label)
    if sample_shard >= 0:
        destination = destination / f"sample-shard-{sample_shard}"
    destination.mkdir(parents=True, exist_ok=False)
    adapter = Path(STORE, "outputs", adapter_name)
    if not (adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"Missing adapter {adapter}")
    try:
        conditions = (("sft-think", 8),) if sample_shard >= 0 else (
            ("plain", 1), ("sft-think", 1))
        for style, count in conditions:
            offset = max(0, sample_shard) * 192
            limit = 192 if sample_shard >= 0 else 0
            command = [sys.executable, "-m", "scripts.evaluate",
                       "--data", f"{STORE}/data/gsm8k", "--split", "validation",
                       "--max-samples", str(limit), "--question-offset", str(offset),
                       "--max-new-tokens", "512", "--batch-size", "64",
                       "--adapter", str(adapter), "--prompt-style", style,
                       "--samples-per-question", str(count), "--sample-seed", str(42 + max(0, sample_shard)),
                       "--out", str(destination / f"{style}-{count}.json")]
            if style == "plain":
                command.extend(["--expected-dataset-sha256", CONTROL_HASH])
            with (destination / "commands.jsonl").open("a") as log:
                log.write(json.dumps(command) + "\n")
            print(f"{label}: {style}, {count} completion(s) per question, offset {offset}", flush=True)
            subprocess.run(command, cwd="/project", check=True)
            volume.commit()
    finally:
        volume.commit()
    return str(destination)


@app.local_entrypoint()
def main(run_name: str, sample_shards: bool = False):
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Invalid run name")
    jobs = [(run_name, "original", BASELINE),
            *[(run_name, f"step{step}", f"{CONTINUATION}/adapter/checkpoint-{step}")
              for step in (50, 100)]]
    if sample_shards:
        jobs = [(*job, shard) for job in jobs for shard in range(4)]
    for result in evaluate.starmap(jobs, order_outputs=False):
        print(result, flush=True)
