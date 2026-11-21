"""Stage 1: LoRA SFT of Qwen3-0.6B-Base on gold GSM8K solutions.

On a CUDA host with a matching torchao/PyTorch installation:
    python -m scripts.train_sft --precision fp8 --smoke-test
    python -m scripts.train_sft --precision fp8 --steps 300

FP8 uses torchao dynamic scaling for frozen LoRA base projections. Adapter
operations, embeddings and the output head keep higher precision. BF16 autocast
is the surrounding compute policy; no separate Accelerate FP8 conversion is
needed. Saved adapters can be loaded by the existing higher-precision evaluator.
"""

import argparse
from contextlib import nullcontext
import json
import math
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model_state_dict
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from scripts.data import MODEL_ID, as_sft_dataset, extract_final


def audit_truncation(dataset, tokenizer, max_length: int) -> dict:
    """Check raw gold targets using the same plain-text prompt/completion format."""
    truncated = lost_final = 0
    longest = 0
    for row in dataset:
        text = row["prompt"] + row["completion"]
        if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        ids = tokenizer(text, truncation=False)["input_ids"]
        longest = max(longest, len(ids))
        truncated += len(ids) > max_length
        retained = tokenizer.decode(ids[:max_length], skip_special_tokens=True)
        lost_final += extract_final(retained) != extract_final(row["completion"])
    return {"examples": len(dataset), "truncated": truncated,
            "lost_final_answer": lost_final, "longest_tokens": longest,
            "max_length": max_length}


def fp8_layer_filter(module: torch.nn.Module, name: str) -> bool:
    """Convert frozen LoRA base projections only; preserve adapters and output head."""
    return (
        isinstance(module, torch.nn.Linear)
        and name.endswith(".base_layer")
        and not module.weight.requires_grad
        and module.in_features % 16 == 0
        and module.out_features % 16 == 0
    )


def enable_fp8(model) -> list[str]:
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as exc:
        raise RuntimeError(
            "FP8 requires a torchao build compatible with your CUDA PyTorch version. "
            "Install it in the GPU environment, not on the Mac."
        ) from exc
    names = [name for name, module in model.named_modules() if fp8_layer_filter(module, name)]
    if not names:
        raise RuntimeError("No eligible frozen LoRA base projections found for FP8")
    original_parameters = {name: model.get_submodule(name).weight for name in names}
    convert_to_float8_training(
        model,
        module_filter_fn=fp8_layer_filter,
        config=Float8LinearConfig(pad_inner_dim=True, emulate=False),
    )
    # Conversion must preserve the exact Parameter objects, including frozen flags.
    if any(model.get_submodule(name).weight is not original_parameters[name]
           or model.get_submodule(name).weight.requires_grad for name in names):
        raise RuntimeError("FP8 conversion changed base Parameter identity or frozen flags")
    print(f"Enabled torchao FP8 compute on {len(names)} frozen base projections; "
          "LoRA adapters and other operations retain higher precision.")
    return names


class CheckTraining(TrainerCallback):
    """Stop on nonfinite gradients/loss instead of silently completing a bad pilot."""

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        # One host synchronization per optimizer step, rather than one per tensor.
        if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item():
            raise RuntimeError("Missing or nonfinite adapter gradients")

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "eval_loss", "grad_norm"):
            if key in (logs or {}) and not math.isfinite(logs[key]):
                raise RuntimeError(f"Nonfinite {key}: {logs[key]}")


class StepProfiler(TrainerCallback):
    def __init__(self, profiler):
        self.profiler = profiler

    def on_substep_end(self, args, state, control, **kwargs):
        self.profiler.step()

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gsm8k")
    parser.add_argument("--output", default="outputs/sft")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=16, help="Examples per GPU microbatch")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Trade compute for lower activation memory")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp8"], default="auto")
    parser.add_argument("--smoke-test", action="store_true", help="Three optimizer steps on a small subset")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if min(args.eval_steps, args.logging_steps, args.save_steps) < 1:
        parser.error("eval-steps, logging-steps and save-steps must be positive")
    if min(args.batch_size, args.eval_batch_size, args.max_length, args.gradient_accumulation_steps) < 1:
        parser.error("batch sizes, max-length and accumulation must be positive")
    if args.smoke_test:
        args.steps = 3
        if args.output == "outputs/sft":
            args.output = "outputs/sft-smoke"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The initial reproducible training path requires an NVIDIA CUDA GPU"
        )
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if precision in ("bf16", "fp8") and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{precision} requires BF16-capable hardware; use --precision fp16 on T4")
    if precision == "fp8" and torch.cuda.get_device_capability() < (8, 9):
        raise RuntimeError("Native FP8 requires compute capability >= 8.9, such as L4 or H100")
    train_ds = as_sft_dataset(load_from_disk(str(Path(args.data) / "train")))
    val_ds = as_sft_dataset(load_from_disk(str(Path(args.data) / "validation")))
    if args.smoke_test:
        train_ds = train_ds.select(range(min(48, len(train_ds))))
        val_ds = val_ds.select(range(min(8, len(val_ds))))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    truncation = {name: audit_truncation(ds, tokenizer, args.max_length)
                  for name, ds in (("train", train_ds), ("validation", val_ds))}
    print(f"Truncation audit: {json.dumps(truncation)}", flush=True)
    # Microbatch size and accumulation together determine examples per update.
    # Completion-only loss teaches gold solutions while masking prompt tokens.
    cfg = SFTConfig(
        output_dir=args.output,
        learning_rate=1e-4,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        max_length=args.max_length,
        pad_to_multiple_of=16 if precision == "fp8" else None,
        padding_free=False,
        completion_only_loss=True,
        eval_strategy="steps",
        eval_steps=3 if args.smoke_test else args.eval_steps,
        eval_on_start=args.eval_on_start,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=1 if args.smoke_test else args.logging_steps,
        logging_nan_inf_filter=False,
        bf16=precision in ("bf16", "fp8"),
        fp16=precision == "fp16",
        report_to="none",
        seed=42,
        model_init_kwargs={"dtype": torch.bfloat16} if precision == "fp8" else None,
    )
    # Rank is the adapter bottleneck dimension, independent of numerical precision.
    adapter = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear"
    )
    trainer = SFTTrainer(
        model=MODEL_ID,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=adapter,
        callbacks=[CheckTraining()],
        processing_class=tokenizer,
    )
    # Attach LoRA first, then convert only its frozen base projections to FP8.
    fp8_layers = enable_fp8(trainer.model) if precision == "fp8" else []
    effective_batch = args.batch_size * args.gradient_accumulation_steps * trainer.args.world_size
    print(f"Microbatch={args.batch_size}, accumulation={args.gradient_accumulation_steps}, "
          f"effective batch={effective_batch}, checkpointing={args.gradient_checkpointing}", flush=True)
    # A successful backward pass alone does not prove the adapters were updated.
    before = ({name: value.detach().cpu().clone()
               for name, value in get_peft_model_state_dict(trainer.model).items()}
              if args.smoke_test else None)
    # Record CPU dispatch of FP8 GEMMs for one microbatch only. Capturing CUDA
    # events for the entire pilot can take minutes to finalize under virtualization.
    profiling = args.smoke_test and precision == "fp8"
    context = (torch.profiler.profile(
                   activities=[torch.profiler.ProfilerActivity.CPU],
                   schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                   acc_events=True,
               )
               if profiling else nullcontext())
    torch.cuda.reset_peak_memory_stats()
    with context as profiler:
        if profiling:
            trainer.add_callback(StepProfiler(profiler))
        train_result = trainer.train()
        if profiling:
            print("Finalizing the one-microbatch FP8 dispatch check...", flush=True)
    fp8_matmul_calls = None
    if profiling:
        fp8_matmul_calls = sum(event.count for event in profiler.key_averages()
                               if "_scaled_mm" in event.key)
        if not fp8_matmul_calls:
            raise RuntimeError("FP8 smoke test recorded no scaled matrix multiplications")
        print(f"FP8 smoke test recorded {fp8_matmul_calls} scaled matrix multiplication events")
    if before is not None:
        after = get_peft_model_state_dict(trainer.model)
        if not any(not torch.equal(before[name], value.detach().cpu()) for name, value in after.items()):
            raise RuntimeError("Smoke test failed: no adapter weights changed")
    trainer.save_model(args.output)
    trainer.save_state()  # Persist final log history even between checkpoint steps.
    metadata = {
        "precision": precision,
        "fp8_backend": "torchao" if fp8_layers else None,
        "fp8_layers": fp8_layers,
        "smoke_scaled_mm_events": fp8_matmul_calls,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "steps": args.steps,
        "eval_steps": cfg.eval_steps,
        "eval_on_start": args.eval_on_start,
        "logging_steps": cfg.logging_steps,
        "save_steps": args.save_steps,
        "smoke_test": args.smoke_test,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "truncation": truncation,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch,
        "gradient_checkpointing": args.gradient_checkpointing,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "training_metrics": train_result.metrics,
        "packages": {name: version(name) for name in
                     ["torch", "transformers", "trl", "peft", "accelerate"]
                     + (["torchao"] if fp8_layers else [])},
    }
    Path(args.output, "training_precision.json").write_text(json.dumps(metadata, indent=2))
    print(f"Peak GPU memory: {metadata['peak_gpu_allocated_gib']:.2f} GiB allocated, "
          f"{metadata['peak_gpu_reserved_gib']:.2f} GiB reserved", flush=True)
    print(f"SFT adapter written to {args.output}")


if __name__ == "__main__":
    main()
