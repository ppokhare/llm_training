"""Replay the saved failing microbatch without generation or optimizer updates."""
import modal
from scripts.modal_sft import image, volume, STORE

app = modal.App('qwen3-grpo-diagnosis')
RUN = f'{STORE}/outputs/grpo-pilot-20261121-021228-28cae4'


@app.function(image=image, gpu='H100', volumes={STORE:volume}, cpu=2, memory=16384,
              timeout=660, retries=0, scaledown_window=2)
def replay(fix_only: bool = False):
    import inspect
    import json
    import time
    import traceback
    from pathlib import Path
    from importlib.metadata import version
    import torch
    from datasets import load_from_disk
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    from trl import GRPOConfig
    from scripts.train_grpo import CheckedGRPOTrainer, check_adapter_gradients, monitored_reward
    from scripts.data import MODEL_ID, as_rl_dataset

    volume.reload()
    data = torch.load(f'{RUN}/failed_microbatch.pt', map_location='cuda', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    report = dict(packages={n:version(n) for n in ['torch','transformers','trl','peft']}, cases=[])
    print('PACKAGES', report['packages'], flush=True)
    report['trl_logprob_source'] = inspect.getsource(CheckedGRPOTrainer._get_per_token_logps_and_entropies)
    cases = ([('sdpa',True,False)] * 3 if fix_only else
             [('sdpa',True,True),('sdpa',False,True),('eager',False,True)])
    for attention, checkpointing, cudnn in cases:
        torch.backends.cuda.enable_cudnn_sdp(cudnn)
        started = time.monotonic()
        model = AutoPeftModelForCausalLM.from_pretrained(
            f'{RUN}/recovered_adapter', is_trainable=True, dtype=torch.bfloat16,
            attn_implementation=attention).to('cuda')
        cfg = GRPOConfig(output_dir='/tmp/grpo-replay', num_generations=8,
            per_device_train_batch_size=4, gradient_accumulation_steps=8,
            generation_batch_size=32, gradient_checkpointing=checkpointing,
            max_completion_length=512, beta=0.0, loss_type='dapo', scale_rewards='group',
            mask_truncated_completions=True, bf16=True, report_to='none')
        trainer = CheckedGRPOTrainer(model=model, args=cfg, processing_class=tokenizer,
            train_dataset=as_rl_dataset(load_from_disk(f'{STORE}/data/gsm8k/train').select(range(4))),
            reward_funcs=monitored_reward)
        trainer.current_gradient_accumulation_steps=8
        if checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model.train()
        case = dict(attention=attention, checkpointing=checkpointing, cudnn_sdp_enabled=torch.backends.cuda.cudnn_sdp_enabled())
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler, torch.autograd.detect_anomaly(check_nan=True), torch.autocast('cuda',dtype=torch.bfloat16):
                loss = trainer._compute_loss(model, data)
                case['loss']=loss.item()
                loss.backward()
            case['attention_ops'] = [event.key for event in profiler.key_averages() if 'scaled_dot_product' in event.key.lower()]
            case['gradients'] = check_adapter_gradients(model)
            case['passed']=True
        except Exception as exc:
            case.update(passed=False,error=str(exc),traceback=traceback.format_exc())
        case['seconds']=time.monotonic()-started
        report['cases'].append(case)
        print('CASE',json.dumps(case),flush=True)
        del trainer, model
        import gc
        gc.collect(); torch.cuda.empty_cache()
        Path(f'{RUN}/diagnosis-fixed.json' if fix_only else f'{RUN}/diagnosis.json').write_text(json.dumps(report,indent=2))
        volume.commit()
    return report


@app.local_entrypoint()
def main(fix_only: bool = False):
    import json
    from pathlib import Path
    result = replay.remote(fix_only)
    Path('results/grpo-diagnosis-fixed.json' if fix_only else 'results/grpo-diagnosis.json').write_text(json.dumps(result,indent=2))
    print('Saved replay diagnostics (fixed backend)' if fix_only else 'Saved replay diagnostics (baseline)')
