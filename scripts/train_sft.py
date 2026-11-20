"""Stage 1: LoRA SFT of Qwen3-0.6B-Base on gold GSM8K solutions."""

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from scripts.data import MODEL_ID, as_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gsm8k")
    parser.add_argument("--output", default="outputs/sft")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The initial reproducible training path requires an NVIDIA CUDA GPU"
        )
    train_ds = as_sft_dataset(load_from_disk(str(Path(args.data) / "train")))
    val_ds = as_sft_dataset(load_from_disk(str(Path(args.data) / "validation")))
    cfg = SFTConfig(
        output_dir=args.output,
        learning_rate=1e-4,
        max_steps=args.steps,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_length=512,
        completion_only_loss=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=42,
    )
    adapter = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear"
    )
    trainer = SFTTrainer(
        model=MODEL_ID,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=adapter,
    )
    trainer.train()
    trainer.save_model(args.output)
    trainer.save_state()  # Persist final log history even between checkpoint steps.
    print(f"SFT adapter written to {args.output}")


if __name__ == "__main__":
    main()
