"""Train validation-gated BF16/FP8 stages from the verified step-100 checkpoint."""
from pathlib import Path

import modal

from scripts.modal_sft import STORE, image, volume
from scripts.modal_validate_thinking import CONTINUATION

app = modal.App("qwen3-thinking-sft-extension")


@app.function(image=image, gpu="H100", volumes={STORE: volume}, cpu=4, memory=24576,
              timeout=3600, retries=0, scaledown_window=2)
def train(run_name: str, checkpoint: str, step: int):
    import subprocess
    import sys

    volume.reload()
    output = Path(STORE, "outputs", run_name, f"stage-{step}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "scripts.train_sft",
        "--data", f"{STORE}/data/gsm8k", "--output", str(output),
        "--thinking-format", "--train-thinking-tokens", "--precision", "fp8",
        "--max-length", "1024", "--batch-size", "16", "--gradient-accumulation-steps", "2",
        "--eval-batch-size", "16", "--no-eval-on-start", "--learning-rate", "1.5e-4",
        "--steps", "300", "--eval-at", str(step), "--pause-at-update", str(step),
        "--skip-panel-generation", "--resume-thinking-checkpoint", checkpoint,
        "--verify-thinking-fp8",
    ]
    try:
        subprocess.run(command, cwd="/project", check=True)
        next_checkpoint = output / f"checkpoint-{step}"
        if not (next_checkpoint / "optimizer.pt").is_file():
            raise RuntimeError(f"Missing optimizer checkpoint at step {step}")
        return str(next_checkpoint)
    finally:
        volume.commit()


@app.function(image=image, gpu="H100", volumes={STORE: volume}, cpu=3, memory=16384,
              timeout=1800, retries=0, scaledown_window=2)
def evaluate(run_name: str, checkpoint: str, step: int, style: str):
    import json
    import subprocess
    import sys

    volume.reload()
    output = Path(STORE, "results", run_name, f"step{step}-{style}.json")
    if output.exists():
        raise RuntimeError(f"Evaluation already exists: {output}")
    try:
        subprocess.run([
            sys.executable, "-m", "scripts.evaluate", "--split", "validation",
            "--data", f"{STORE}/data/gsm8k", "--adapter", checkpoint,
            "--prompt-style", style, "--max-new-tokens", "512", "--max-samples", "0",
            "--batch-size", "64", "--out", str(output),
        ], cwd="/project", check=True)
        return json.loads(output.read_text())
    finally:
        volume.commit()


@app.local_entrypoint()
def main(run_name: str, baseline_dir: str):
    import json
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Invalid run name")
    styles = ("plain", "sft-think")
    previous = {style: json.loads(Path(baseline_dir, "step100", f"{style}-1.json").read_text())
                for style in styles}
    checkpoint = f"{STORE}/outputs/{CONTINUATION}/adapter/checkpoint-100"
    decisions = Path("results", f"{run_name}-decisions.jsonl")
    with decisions.open("x") as log:
        for step in (150, 200, 250, 300):
            checkpoint = train.remote(run_name, checkpoint, step)
            current = dict(zip(styles, evaluate.starmap(
                [(run_name, checkpoint, step, style) for style in styles])))
            for style in styles:
                fields = ("dataset_sha256", "batch_size", "n", "max_new_tokens", "do_sample")
                if any(current[style][key] != previous[style][key] for key in fields):
                    raise RuntimeError(f"Validation settings changed for {style}")
            thinking, prior = current["sft-think"], previous["sft-think"]
            gain = thinking["accuracy_greedy"] - prior["accuracy_greedy"]
            format_drop = prior["think_format_rate"] - thinking["think_format_rate"]
            plain_drop = previous["plain"]["accuracy_greedy"] - current["plain"]["accuracy_greedy"]
            continue_training = gain > 0 and format_drop <= .01 and plain_drop <= .03 and step < 300
            entry = {"step": step, "checkpoint": checkpoint,
                     "thinking_accuracy_change": gain, "thinking_format_drop": format_drop,
                     "plain_accuracy_drop": plain_drop, "continue": continue_training}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)
            if not continue_training:
                break
            previous = current
