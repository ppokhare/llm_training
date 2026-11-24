"""Bounded BF16 GRPO continuation of an immutable SFT LoRA checkpoint."""

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import AutoPeftModelForCausalLM, get_peft_model_state_dict
from transformers import AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from scripts.data import MODEL_ID, as_rl_dataset, reward_exact_match, extract_final, FINAL_RE, with_prompt_style, make_prompt, has_think_format
from scripts.train_sft import CheckTraining


def frozen_hash(model):
    """Hash frozen parameters so the smoke test detects unintended base updates."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class GradientFailure(RuntimeError):
    """An invalid backward pass detected before the optimizer can update weights."""


def check_adapter_gradients(model):
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    present = [(name, p.grad) for name, p in parameters if p.grad is not None]
    missing = [name for name, p in parameters if p.grad is None]
    finite = torch.stack([torch.isfinite(g).all() for _, g in present]).tolist() if present else []
    bad = [name for (name, _), ok in zip(present, finite) if not ok]
    report = dict(trainable_tensors=len(parameters), gradient_tensors=len(present),
                  missing_gradient_names=missing, nonfinite_gradient_names=bad)
    if not present or bad:
        raise GradientFailure(json.dumps(report))
    return report


class CheckedGRPOTrainer(GRPOTrainer):
    """Catch bad backward passes before clipping and retain their actual inputs."""
    def _compute_loss(self, model, inputs):
        self.last_loss_inputs = inputs
        loss = super()._compute_loss(model, inputs)
        if not torch.isfinite(loss.detach()).all().item():
            raise GradientFailure('Nonfinite GRPO loss before backward')
        return loss

    def training_step(self, model, inputs, num_items_in_batch):
        result = super().training_step(model, inputs, num_items_in_batch)
        check_adapter_gradients(model)
        return result

    def _clip_grad_norm(self, model):
        norm = super()._clip_grad_norm(model)
        if not torch.isfinite(torch.as_tensor(norm)).all().item():
            raise GradientFailure(f'Nonfinite gradient norm during clipping: {norm}')
        check_adapter_gradients(model)
        return norm


class RolloutGuard(TrainerCallback):
    """Stop sustained degenerate rollouts, while retaining Trainer diagnostics."""
    def __init__(self):
        self.bad = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        bad = (logs.get('frac_reward_zero_std', 0) >= 1.0
               or logs.get('completions/clipped_ratio', 0) > 0.25
               or logs.get('verifier/invalid_fraction', 0) > 0.25)
        self.bad = self.bad + 1 if bad else 0
        if self.bad >= 5:
            raise RuntimeError('Five consecutive degenerate rollout updates; inspect diversity/length')


def monitored_reward(completions, answer, **kwargs):
    rewards = reward_exact_match(completions, answer)
    if log_metric := kwargs.get('log_metric'):
        log_metric('verifier/think_format_fraction', sum(has_think_format(c) for c in completions)/len(completions))
        log_metric('verifier/invalid_fraction', sum(extract_final(c) is None for c in completions)/len(completions))
        log_metric('verifier/multiple_tags_fraction', sum(len(FINAL_RE.findall(c)) > 1 for c in completions)/len(completions))
        # These are diagnostic metrics only; the binary correctness reward is unchanged.
        token_ids = kwargs.get('completion_ids')
        if token_ids is not None and len(token_ids) == len(rewards):
            for correct, label in [(True, 'correct'), (False, 'incorrect')]:
                lengths = [len(ids) for ids, reward in zip(token_ids, rewards) if bool(reward) == correct]
                if lengths:
                    log_metric(f'verifier/{label}_mean_tokens', sum(lengths)/len(lengths))
        groups = {}
        for prompt, reward in zip(kwargs.get('prompts', []), rewards):
            groups.setdefault(prompt, []).append(reward)
        if groups:
            log_metric('verifier/all_correct_group_fraction', sum(all(g) for g in groups.values())/len(groups))
            log_metric('verifier/all_wrong_group_fraction', sum(not any(g) for g in groups.values())/len(groups))
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/store/data/gsm8k')
    parser.add_argument('--sft', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['diversity', 'smoke', 'pilot'], required=True)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--num-generations', type=int, default=4)
    parser.add_argument('--attention', choices=['sdpa', 'flash'], default='sdpa')
    parser.add_argument('--prompt-style', choices=['plain', 'think'], default='think')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.num_generations < 2:
        parser.error('--num-generations must be at least 2')
    generations = args.num_generations
    generation_batch = 4 * generations  # Preserve four distinct questions per update.
    out = Path(args.output)
    if out.exists() or out.resolve() == Path(args.sft).resolve():
        raise RuntimeError('Output must be a new directory separate from SFT')
    out.mkdir(parents=True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This path requires a BF16 CUDA GPU')
    # The saved padded BF16 rollout reproduces NaNs in cuDNN SDPA backward.
    # Leave other SDPA backends available; do not disable all cuDNN operations.
    torch.backends.cuda.enable_cudnn_sdp(False)
    attention = 'kernels-community/flash-attn2' if args.attention == 'flash' else 'sdpa'
    print(f'Attention: {attention}; cuDNN SDPA disabled', flush=True)
    set_seed(42)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.sft, is_trainable=True, dtype=torch.bfloat16, attn_implementation=attention)
    model.to('cuda')
    train = with_prompt_style(load_from_disk(str(Path(args.data) / 'train')), args.prompt_style)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name for name in trainable):
        raise RuntimeError('Expected only existing LoRA parameters to be trainable')
    metadata = dict(mode=args.mode, sft=args.sft, precision='bf16', trainable=trainable,
                    prompt_style=args.prompt_style, prompt_template=make_prompt('{question}', args.prompt_style),
                    attention_implementation=attention, cudnn_sdp_enabled=torch.backends.cuda.cudnn_sdp_enabled(),
                    cudnn_version=torch.backends.cudnn.version(),
                    num_generations=generations, generation_batch_size=generation_batch,
                    prompts_per_update=4, gradient_accumulation_steps=generations,
                    packages={n: version(n) for n in ['torch', 'transformers', 'trl', 'peft', 'accelerate']
                              + (['kernels'] if args.attention == 'flash' else [])},
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), Path(__file__).with_name('data.py')]},
                    starting_adapter_sha256=hashlib.sha256(
                        Path(args.sft, 'adapter_model.safetensors').read_bytes()).hexdigest())
    if args.mode == 'diversity':
        model.eval()
        records = []
        # Shuffle once, then preserve IDs and raw samples for an auditable fixed probe.
        probe = train.shuffle(seed=42).select(range(128))
        for start in range(0, len(probe), 4):
            rows = [probe[i] for i in range(start, min(start + 4, len(probe)))]
            inputs = tokenizer([r['prompt'] for r in rows], padding=True, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                outputs = model.generate(**inputs, do_sample=True, temperature=1.0, top_p=1.0,
                                         top_k=0, num_return_sequences=generations, max_new_tokens=512,
                                         use_cache=True, pad_token_id=tokenizer.pad_token_id)
            for i, row in enumerate(rows):
                samples = []
                for output in outputs[i*generations:(i+1)*generations]:
                    ids = output[inputs['input_ids'].shape[1]:].tolist()
                    eos = model.generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else eos or []
                    end = next((j+1 for j, token in enumerate(ids) if token in eos), len(ids))
                    ids = ids[:end]
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    samples.append(dict(output=text, output_with_special_tokens=tokenizer.decode(ids, skip_special_tokens=False),
                                        generated_token_ids=ids, generated_think_token_ids=[t for t in ids if t in set(tokenizer.convert_tokens_to_ids(['<think>', '</think>']))], reward=reward_exact_match([text], [row['answer']])[0],
                                        tokens=len(ids), truncated=len(ids) >= 512 and not any(t in eos for t in ids),
                                        invalid=extract_final(text) is None, think_format=has_think_format(text), tag_count=len(FINAL_RE.findall(text))))
                records.append(dict(id=row['id'], samples=samples))
            (out/'samples.partial.json').write_text(json.dumps(records))
            print(f'Diversity {len(records)}/128', flush=True)
        totals = [sum(s['reward'] for s in r['samples']) for r in records]
        samples = [s for r in records for s in r['samples']]
        metadata.update(mixed_fraction=sum(0 < n < generations for n in totals)/128,
                        all_correct_fraction=totals.count(generations)/128, all_wrong_fraction=totals.count(0)/128,
                        accuracy=sum(s['reward'] for s in samples)/len(samples),
                        invalid_fraction=sum(s['invalid'] for s in samples)/len(samples),
                        think_format_fraction=sum(s['think_format'] for s in samples)/len(samples),
                        truncated_fraction=sum(s['truncated'] for s in samples)/len(samples), records=records)
    else:
        before = {n: t.detach().cpu().clone() for n, t in get_peft_model_state_dict(model).items()}
        base_before = frozen_hash(model) if args.mode == 'smoke' else None
        cfg = GRPOConfig(
            output_dir=str(out), learning_rate=1e-5, max_steps=3 if args.mode == 'smoke' else args.steps,
            num_generations=generations, max_completion_length=512, temperature=1.0, top_p=1.0, top_k=0,
            per_device_train_batch_size=4, gradient_accumulation_steps=generations,
            generation_batch_size=generation_batch, gradient_checkpointing=True,
            beta=0.0, use_vllm=False, loss_type='dapo', scale_rewards='group',
            mask_truncated_completions=True, num_iterations=1,
            logging_steps=1, logging_nan_inf_filter=False,
            save_strategy='steps', save_steps=10, bf16=True, fp16=False,
            report_to='none', seed=42)
        trainer = CheckedGRPOTrainer(model=model, args=cfg, train_dataset=as_rl_dataset(train),
                              processing_class=tokenizer, reward_funcs=monitored_reward,
                              callbacks=[CheckTraining(), RolloutGuard()])
        torch.cuda.reset_peak_memory_stats()
        try:
            result = trainer.train()
        except GradientFailure as exc:
            # No optimizer update has occurred on this invalid backward pass.
            # Preserve last good weights, but do not advertise a resumable optimizer checkpoint.
            failure = dict(error=str(exc), completed_steps=trainer.state.global_step,
                           failed_update=trainer.state.global_step + 1)
            (out/'failure.json').write_text(json.dumps(failure, indent=2))
            def cpu_copy(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu()
                if isinstance(value, dict):
                    return {k: cpu_copy(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [cpu_copy(v) for v in value]
                return value
            torch.save(cpu_copy(getattr(trainer, 'last_loss_inputs', {})), out/'failed_microbatch.pt')
            adapter = get_peft_model_state_dict(model)
            if all(torch.isfinite(t).all().item() for t in adapter.values()):
                trainer.save_model(str(out/'recovered_adapter'))
            trainer.save_state()
            raise
        except BaseException:
            trainer.save_state()
            raise
        after = get_peft_model_state_dict(model)
        changed = [n for n, t in after.items() if not torch.equal(before[n], t.detach().cpu())]
        if not changed:
            raise RuntimeError('No adapter weights changed')
        if base_before is not None and base_before != frozen_hash(model):
            raise RuntimeError('Frozen base parameters changed')
        trainer.save_model(str(out))
        trainer.save_state()
        metadata.update(changed_adapter_tensors=len(changed), frozen_base_unchanged=True if base_before else None,
                        training_metrics=result.metrics, config=cfg.to_dict(),
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    metadata['elapsed_seconds'] = time.monotonic()-started
    (out/'audit.json').write_text(json.dumps(metadata, indent=2, default=str))
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['records','trainable','config']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
