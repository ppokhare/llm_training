"""Run a gated fresh thinking-format SFT on the original torchao stack."""
from pathlib import Path

import modal

from scripts.modal_sft import STORE, image, volume

app = modal.App("qwen3-gsm8k-fresh-thinking-fp8")
BASELINE = f"{STORE}/outputs/sft-20261121-002201-0e7388"


@app.function(image=image, gpu="H100", volumes={STORE: volume}, cpu=4, memory=24576,
              timeout=3600, retries=0, scaledown_window=2)
def train(run_name: str, stage: str):
    steps, eval_at, samples = {
        "verification": (10, "10", 16),
        "pilot25": (25, "25", 64),
        "pilot": (32, "32", 32),
        "full": (420, "105,210,315,420", 0),
        "resume": (100, "25,50,100", 64),
    }[stage]
    import json
    import subprocess
    import sys

    volume.reload()
    output = Path(STORE, "outputs", run_name)
    data = Path(STORE, "data", "gsm8k")
    if output.exists() or not all((data / split).is_dir() for split in ("train", "validation")):
        raise RuntimeError("Expected a new output directory and the existing development splits")
    output.mkdir()
    adapter = output / "adapter"

    def run(*args):
        command = [sys.executable, "-m", *args]
        with (output / "commands.jsonl").open("a") as commands:
            commands.write(json.dumps(command) + "\n")
        with (output / "run.log").open("a") as log:
            with subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)

    def evaluate(checkpoint, label, style):
        result = output / f"{label}-{style}.json"
        run("scripts.evaluate", "--data", str(data), "--adapter", str(checkpoint),
            "--split", "validation", "--prompt-style", style,
            "--max-samples", str(samples), "--max-new-tokens", "512",
            "--batch-size", "16", "--out", str(result))
        return json.loads(result.read_text())

    try:
        resume_args = []
        if stage == "resume":
            checkpoint = Path(STORE, "outputs", "sft-think-fp8-resume-20261123-222025-verification",
                              "adapter", "checkpoint-10")
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
                if not (checkpoint / name).is_file():
                    raise RuntimeError(f"Missing step-10 resume artifact {name}")
            resume_args = ["--resume-thinking-checkpoint", str(checkpoint)]
        run("scripts.train_sft", "--data", str(data), "--output", str(adapter),
            "--thinking-format", "--train-thinking-tokens", "--precision", "fp8",
            "--max-length", "1024", "--learning-rate", "3e-4", "--batch-size", "16",
            "--gradient-accumulation-steps", "2", "--eval-batch-size", "16",
            "--no-eval-on-start", "--steps", str(steps),
            "--eval-at", eval_at,
            *(["--verify-thinking-fp8"] if stage in ("verification", "pilot", "resume") else []),
            *resume_args)
        if stage == "resume":
            return str(output)
        thinking = evaluate(adapter, "fresh", "sft-think")
        metrics = {
            "stage": stage, "steps": steps, "n": thinking["n"],
            "think_format_rate": thinking["think_format_rate"],
            "final_valid_rate": 1 - thinking["invalid_format_rate"],
            "accuracy": thinking["accuracy_greedy"],
            "token_limit_rate": thinking["token_limit_rate"],
            "mean_output_tokens": thinking["mean_output_tokens"],
        }
        print(json.dumps(metrics), flush=True)
        (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
        if thinking["think_format_rate"] < .90 or thinking["invalid_format_rate"] > .10:
            raise RuntimeError("Generation gate failed: require ≥90% think compliance and final validity")
        if stage == "full":
            for style in ("plain", "sft-think"):
                fresh = thinking if style == "sft-think" else evaluate(adapter, "fresh", style)
                baseline = evaluate(BASELINE, "original", style)
                if fresh["dataset_sha256"] != baseline["dataset_sha256"]:
                    raise RuntimeError("Matched evaluation prompts differ")
            run("scripts.report_thinking_fp8", str(output))
        (output / "passed.json").write_text(json.dumps(metrics, indent=2))
    finally:
        volume.commit()
    return str(output)


@app.local_entrypoint()
def main(run_name: str, staged_pilots: bool = False, resume_step10: bool = False):
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Use a run name containing just letters, numbers, hyphens and underscores")
    if resume_step10 and staged_pilots:
        raise ValueError("Select only one launch mode")
    stages = ("verification", "pilot25") if staged_pilots else ("pilot", "full")
    if resume_step10:
        stages = ("resume",)
    for stage in stages:
        print(train.remote(f"{run_name}-{stage}", stage), flush=True)
