"""Stage 2B: OFF-POLICY RESPONSE distillation, not KL/logit distillation.

The JSONL must contain verified `prompt` and `completion` strings from a
stronger teacher on examples drawn ONLY from the training partition.
"""

import argparse

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from trl import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--sft", default="outputs/sft")
    parser.add_argument("--output", default="outputs/distilled")
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This starter requires an NVIDIA CUDA GPU for training")
    data = load_dataset("json", data_files=args.teacher_jsonl, split="train")
    if not {"prompt", "completion"}.issubset(data.column_names):
        raise ValueError("Teacher JSONL must contain prompt and completion")
    data = data.select_columns(["prompt", "completion"])
    model = AutoPeftModelForCausalLM.from_pretrained(args.sft, is_trainable=True, torch_dtype="auto")
    cfg = SFTConfig(
        output_dir=args.output, max_steps=args.steps,
        learning_rate=5e-5, max_length=512, completion_only_loss=True,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        logging_steps=10, save_strategy="steps", save_steps=50,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        report_to="none", seed=42,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=data)
    trainer.train()
    trainer.save_model(args.output)
    trainer.save_state()  # Persist final log history even between checkpoint steps.
    print(f"Response-distillation adapter written to {args.output}")


if __name__ == "__main__":
    main()
