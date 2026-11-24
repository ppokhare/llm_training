"""Bounded FlashAttention replay and three-update GRPO test on one H100."""
from pathlib import Path
import modal
from scripts.modal_sft import ROOT, STORE, volume

app = modal.App('qwen3-grpo-flash-test')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install_from_requirements(str(ROOT/'requirements.txt'))
         .pip_install('kernels==0.18.2')
         .env({'HF_HOME':f'{STORE}/hf','PYTHONPATH':'/project','PYTHONUNBUFFERED':'1'})
         .add_local_dir(ROOT/'scripts',remote_path='/project/scripts',ignore=['**/__pycache__/**','**/*.pyc']))
FAILED=f'{STORE}/outputs/grpo-pilot-20261121-021228-28cae4'
SFT=f'{STORE}/outputs/sft-20261121-002201-0e7388'


@app.function(image=image,gpu='H100',volumes={STORE:volume},cpu=2,memory=16384,
              timeout=600,retries=0,scaledown_window=2)
def test_flash(run_name: str):
    import gc
    import json
    import time
    import traceback
    from contextlib import nullcontext
    from importlib.metadata import version
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from datasets import load_from_disk
    from peft import AutoPeftModelForCausalLM,get_peft_model_state_dict
    from transformers import AutoTokenizer,set_seed
    from trl import GRPOConfig
    from scripts.data import MODEL_ID,as_rl_dataset
    from scripts.train_grpo import CheckedGRPOTrainer,check_adapter_gradients,monitored_reward,RolloutGuard
    from scripts.train_sft import CheckTraining

    volume.reload()
    output=Path(STORE,'outputs',run_name)
    if output.exists(): raise RuntimeError('Output already exists')
    output.mkdir(parents=True)
    set_seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,padding_side='left')
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    data=torch.load(f'{FAILED}/failed_microbatch.pt',map_location='cuda',weights_only=True)
    train=as_rl_dataset(load_from_disk(f'{STORE}/data/gsm8k/train'))
    report=dict(packages={n:version(n) for n in ['torch','transformers','trl','peft','kernels']},cases=[])
    def save():
        (output/'flash_test.json').write_text(json.dumps(report,indent=2,default=str));volume.commit()
    def trainer_for(model,steps):
        cfg=GRPOConfig(output_dir=str(output/'adapter'),max_steps=steps,learning_rate=1e-5,
            num_generations=8,generation_batch_size=32,per_device_train_batch_size=4,
            gradient_accumulation_steps=8,gradient_checkpointing=True,max_completion_length=512,
            temperature=1.,top_p=1.,top_k=0,beta=0.,loss_type='dapo',scale_rewards='group',
            mask_truncated_completions=True,bf16=True,report_to='none',logging_steps=1,
            logging_nan_inf_filter=False,save_strategy='no',seed=42)
        return CheckedGRPOTrainer(model=model,args=cfg,processing_class=tokenizer,
            train_dataset=train,reward_funcs=monitored_reward,callbacks=[CheckTraining(),RolloutGuard()])
    try:
        # Force the built-in Flash backend: fallback would invalidate this test.
        for backend in ['pytorch_flash_only','hub_flash_attention_2']:
            case=dict(backend=backend);started=time.monotonic();model=trainer=None
            try:
                impl='sdpa' if backend=='pytorch_flash_only' else 'kernels-community/flash-attn2'
                model=AutoPeftModelForCausalLM.from_pretrained(f'{FAILED}/recovered_adapter',
                    is_trainable=True,dtype=torch.bfloat16,attn_implementation=impl).to('cuda')
                trainer=trainer_for(model,3);trainer.current_gradient_accumulation_steps=8
                model.gradient_checkpointing_enable();model.enable_input_require_grads();model.train()
                context=sdpa_kernel(SDPBackend.FLASH_ATTENTION) if backend=='pytorch_flash_only' else nullcontext()
                with context,torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler,torch.autograd.detect_anomaly(check_nan=True),torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=trainer._compute_loss(model,data);case['loss']=loss.item();loss.backward()
                case['gradients']=check_adapter_gradients(model)
                case['flash_ops']=[e.key for e in profiler.key_averages() if 'flash' in e.key.lower()]
                if not case['flash_ops']: raise RuntimeError('No FlashAttention operations observed')
                case['passed']=True
            except Exception as exc:
                case.update(passed=False,error=str(exc),traceback=traceback.format_exc())
            case['seconds']=time.monotonic()-started;report['cases'].append(case)
            print('REPLAY',json.dumps(case),flush=True);save()
            del trainer,model;gc.collect();torch.cuda.empty_cache()
        if not report['cases'][-1]['passed']:
            report['smoke_skipped']='FlashAttention-2 replay failed';save();return report
        # A fresh model starts at the original SFT checkpoint, not the recovered RL adapter.
        model=AutoPeftModelForCausalLM.from_pretrained(SFT,is_trainable=True,dtype=torch.bfloat16,
                                                      attn_implementation='kernels-community/flash-attn2').to('cuda')
        trainer=trainer_for(model,3)
        before={n:t.detach().cpu().clone() for n,t in get_peft_model_state_dict(model).items()}
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        result=trainer.train()
        after=get_peft_model_state_dict(model)
        changed=sum(not torch.equal(before[n],t.detach().cpu()) for n,t in after.items())
        if not changed: raise RuntimeError('No adapter tensors changed')
        flash_ops=report['cases'][-1]['flash_ops']
        if model.config._attn_implementation != 'kernels-community/flash-attn2':
            raise RuntimeError('Training attention backend changed unexpectedly')
        trainer.save_model(str(output/'adapter'));trainer.save_state()
        report['smoke']=dict(passed=True,steps=trainer.state.global_step,changed_adapter_tensors=changed,
            seconds=time.monotonic()-started,metrics=result.metrics,flash_ops=flash_ops,
            flash_ops_trace_scope='saved_batch_replay',attention_implementation=model.config._attn_implementation,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            history=trainer.state.log_history)
        # Fresh reload, then a few greedy generations with the same Flash backend.
        del trainer,model;gc.collect();torch.cuda.empty_cache()
        model=AutoPeftModelForCausalLM.from_pretrained(str(output/'adapter'),dtype=torch.bfloat16,
                       attn_implementation='kernels-community/flash-attn2').to('cuda').eval()
        rows=train.select(range(3))
        inputs=tokenizer(list(rows['prompt']),padding=True,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=512,do_sample=False,pad_token_id=tokenizer.pad_token_id)
        report['reload']=dict(passed=True,completions=tokenizer.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True))
        save();return report
    except Exception as exc:
        report['error']=str(exc);report['traceback']=traceback.format_exc();save();raise
    finally:
        volume.commit()


@app.function(image=image,gpu='H100',volumes={STORE:volume},cpu=2,memory=16384,
              timeout=120,retries=0,scaledown_window=2)
def reload_flash(run_name: str):
    """Repeat only save/reload QA without repeating successful training."""
    import json
    import torch
    from datasets import load_from_disk
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    from scripts.data import MODEL_ID
    volume.reload()
    output=Path(STORE,'outputs',run_name)
    report=json.loads((output/'flash_test.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,padding_side='left')
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    model=AutoPeftModelForCausalLM.from_pretrained(str(output/'adapter'),dtype=torch.bfloat16,
        attn_implementation='kernels-community/flash-attn2').to('cuda').eval()
    prompts=list(load_from_disk(f'{STORE}/data/gsm8k/train').select(range(3))['prompt'])
    inputs=tokenizer(prompts,padding=True,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        generated=model.generate(**inputs,max_new_tokens=512,do_sample=False,pad_token_id=tokenizer.pad_token_id)
    report['reload']=dict(passed=True,completions=tokenizer.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True))
    if 'error' in report:
        report['initial_reload_harness_error']=report.pop('error')
        report['initial_reload_harness_traceback']=report.pop('traceback',None)
    (output/'flash_test.json').write_text(json.dumps(report,indent=2));volume.commit()
    return report


@app.local_entrypoint()
def main(reload_run: str = ""):
    import json
    from datetime import datetime,timezone
    from uuid import uuid4
    if reload_run:
        result=reload_flash.remote(reload_run)
        Path('results',f'{reload_run}.json').write_text(json.dumps(result,indent=2,default=str))
        print(f'Reload passed: results/{reload_run}.json')
        return
    name=f'grpo-flash-test-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}'
    result=test_flash.remote(name)
    Path('results',f'{name}.json').write_text(json.dumps(result,indent=2,default=str))
    print(f'FlashAttention evidence: results/{name}.json; Volume outputs/{name}')
