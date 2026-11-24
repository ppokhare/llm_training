"""Bounded H100 GRPO stages; each invocation runs one explicitly selected stage."""

from pathlib import Path
import modal
from scripts.modal_sft import image, volume, STORE

app = modal.App('qwen3-gsm8k-grpo')
SFT = 'sft-20261121-002201-0e7388'


@app.function(image=image, volumes={STORE: volume}, cpu=2, memory=8192,
              timeout=120, retries=0, scaledown_window=2)
def preflight(num_generations: int, attention: str, prompt_style: str, expected_plain_hash: str):
    """Check existing artifacts and configuration on CPU before allocating a GPU."""
    import hashlib
    import json
    from importlib.metadata import version
    from trl import GRPOConfig
    from datasets import load_from_disk
    from scripts.data import canonical_number, with_prompt_style
    volume.reload()
    for file in ['adapter_config.json', 'adapter_model.safetensors']:
        if not Path(STORE, 'outputs', SFT, file).exists():
            raise RuntimeError(f'Missing SFT artifact: {file}')
    train = load_from_disk(f'{STORE}/data/gsm8k/train')
    if len(train) != 6725 or any(canonical_number(a) is None for a in train['answer']):
        raise RuntimeError('Unexpected training data')
    validation = load_from_disk(f'{STORE}/data/gsm8k/validation')
    def digest(dataset):
        return hashlib.sha256(json.dumps(
            [{k:row[k] for k in ('question','prompt','answer')} for row in dataset],
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if digest(validation) != expected_plain_hash:
        raise RuntimeError('Stored validation split differs from the established baseline')
    validation_hash = digest(with_prompt_style(validation, prompt_style))
    cfg = GRPOConfig(output_dir='/tmp/preflight', bf16=False, fp16=False,
                     use_cpu=True, num_generations=num_generations, per_device_train_batch_size=4,
                     gradient_accumulation_steps=num_generations, generation_batch_size=4*num_generations,
                     loss_type='dapo', scale_rewards='group', mask_truncated_completions=True)
    result = dict(packages={n:version(n) for n in ['torch','trl','transformers','peft']},
                  generation_batch_size=cfg.generation_batch_size, num_generations=num_generations, distinct_prompts=4)
    result['attention'] = attention
    result['prompt_style'] = prompt_style
    result['validation_dataset_sha256'] = validation_hash
    if attention == 'flash':
        result['packages']['kernels'] = version('kernels')
    print(result)
    return result


@app.function(image=image, gpu='H100', volumes={STORE:volume}, cpu=2, memory=16384,
              timeout=11100, retries=0, scaledown_window=2)
def run_stage(mode: str, run_name: str, seconds: int, expected_hash: str, steps: int, num_generations: int, attention: str, prompt_style: str):
    """Use one overall subprocess deadline, including fresh-process reload checks."""
    import subprocess
    import sys
    import time
    volume.reload()
    started = time.monotonic()
    output = f'{STORE}/outputs/{run_name}'
    if Path(output).exists():
        raise RuntimeError('Output already exists')
    def run(command):
        remaining = seconds-(time.monotonic()-started)
        if remaining <= 0:
            raise RuntimeError('Stage time budget exhausted')
        subprocess.run([sys.executable, '-m', *command], cwd='/project', check=True, timeout=remaining)
    try:
        run(['scripts.train_grpo', '--mode', mode, '--sft', f'{STORE}/outputs/{SFT}', '--output', output, '--steps', str(steps),
             '--num-generations', str(num_generations), '--attention', attention, '--prompt-style', prompt_style])
        if mode != 'diversity':
            run(['scripts.evaluate', '--data', f'{STORE}/data/gsm8k', '--adapter', output,
                 '--prompt-style',prompt_style,'--split','validation','--max-samples','3' if mode == 'smoke' else '0',
                 '--max-new-tokens','512','--batch-size','3' if mode == 'smoke' else '64',
                 *(['--expected-dataset-sha256', expected_hash] if mode == 'pilot' else []),
                 '--out',f'{output}/validation.json'])
    finally:
        volume.commit()
    return dict(output=output, elapsed_seconds=time.monotonic()-started)


@app.local_entrypoint()
def main(mode: str = 'diversity', seconds: int = 480, steps: int = 25, num_generations: int = 4, attention: str = 'sdpa', prompt_style: str = 'think'):
    import json
    from datetime import datetime, timezone
    from uuid import uuid4
    if mode not in ['diversity','smoke','pilot'] or not 1 <= seconds <= 10800 or steps < 1 or num_generations < 2 or attention not in ['sdpa', 'flash'] or prompt_style not in ['plain', 'think']:
        raise ValueError('Invalid stage or deadline')
    expected = json.loads(Path('results/base_val_full_512.json').read_text())['dataset_sha256']
    prepared = preflight.remote(num_generations, attention, prompt_style, expected)
    expected = prepared['validation_dataset_sha256']
    label = f'full-{attention}' if mode == 'pilot' and steps >= 300 else mode
    name = f'grpo-{label}-{prompt_style}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}'
    print(f'Stage {name}, {steps} requested steps, {num_generations} answers/question, {attention} attention, {prompt_style} prompt, subprocess deadline {seconds}s', flush=True)
    print(run_stage.remote(mode, name, seconds, expected, steps, num_generations, attention, prompt_style))
    print(f'Download: uv run modal volume get qwen3-gsm8k outputs/{name} ./results/{name}')
