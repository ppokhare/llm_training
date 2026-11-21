"""Continue the SFT LoRA adapter with group-relative, verifiable-reward RL.

GRPO samples several answers to each training prompt and compares their rewards.
The shared verifier scores numerical final answers without a learned reward model.
"""

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import AutoPeftModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from scripts.data import as_rl_dataset, reward_exact_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gsm8k")
    parser.add_argument("--sft", default="outputs/sft")
    parser.add_argument("--output", default="outputs/grpo")
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO training needs an NVIDIA CUDA GPU in this starter")
    # Continue the SFT adapter rather than creating a second adapter on top of it.
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.sft, is_trainable=True, torch_dtype="auto"
    )
    model.config.use_cache = False
    # The trainer gets questions and answer keys, but never the gold reasoning.
    train_ds = as_rl_dataset(load_from_disk(str(Path(args.data) / "train")))
    cfg = GRPOConfig(
        output_dir=args.output,
        learning_rate=1e-5,
        max_steps=args.steps,
        num_generations=4,  # Rewards are compared within each prompt's sample group.
        max_completion_length=256,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        beta=0.0,  # No reference-policy KL penalty in this initial configuration.
        use_vllm=False,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=42,
    )
    trainer = GRPOTrainer(
        model=model, args=cfg, train_dataset=train_ds, reward_funcs=reward_exact_match
    )
    trainer.train()
    trainer.save_model(args.output)
    trainer.save_state()  # Persist final log history even between checkpoint steps.
    print(f"GRPO adapter written to {args.output}")


if __name__ == "__main__":
    main()
