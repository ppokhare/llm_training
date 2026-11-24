"""Post-hoc fixed-validation probe of saved GRPO checkpoints; no updates."""
import modal
from scripts.modal_sft import image,volume,STORE

app=modal.App('qwen3-grpo-fixed-probe')
RUN='grpo-full-flash-20261121-031651-9a829c'
SFT='sft-20261121-002201-0e7388'


@app.function(image=image,gpu='H100',volumes={STORE:volume},cpu=2,memory=16384,
              timeout=720,retries=0,scaledown_window=2)
def probe(output_name: str):
    import gc
    import hashlib
    import json
    import time
    from pathlib import Path
    import torch
    from datasets import load_from_disk
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer,set_seed
    from scripts.data import MODEL_ID,extract_final

    volume.reload();started=time.monotonic()
    destination=Path(STORE,'results',output_name)
    if destination.exists():raise RuntimeError('Probe output exists')
    destination.mkdir(parents=True)
    rows=list(load_from_disk(f'{STORE}/data/gsm8k/validation').shuffle(seed=42).select(range(64)))
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,padding_side='left')
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    torch.backends.cuda.enable_cudnn_sdp(False)
    report=dict(run=RUN,split='validation',n=64,selection='shuffle(seed=42), first 64',
        ids=[r['id'] for r in rows],max_new_tokens=512,temperature=1.,top_p=1.,top_k=0,
        sampled_generations_per_question=8,attention='sdpa, cuDNN disabled',hardware='H100',dtype='bfloat16',
        dataset_sha256=hashlib.sha256(json.dumps([{k:r[k] for k in ['id','prompt','answer']} for r in rows],sort_keys=True).encode()).hexdigest(),checkpoints=[])
    def save():
        (destination/'probe.json').write_text(json.dumps(report,indent=2));volume.commit()
    def generate(model,selected,count,sampling):
        inputs=tokenizer([r['prompt'] for r in selected],padding=True,return_tensors='pt').to('cuda')
        kwargs=dict(max_new_tokens=512,do_sample=sampling,num_return_sequences=count,pad_token_id=tokenizer.pad_token_id)
        if sampling:kwargs.update(temperature=1.,top_p=1.,top_k=0)
        with torch.inference_mode():outputs=model.generate(**inputs,**kwargs)
        eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
        records=[]
        for i,row in enumerate(selected):
            for seq in outputs[i*count:(i+1)*count]:
                ids=seq[inputs['input_ids'].shape[1]:].tolist()
                end=next((j+1 for j,t in enumerate(ids) if t in eos),len(ids));ids=ids[:end]
                text=tokenizer.decode(ids,skip_special_tokens=True);pred=extract_final(text)
                records.append(dict(id=row['id'],gold=row['answer'],prediction=pred,correct=pred==row['answer'],
                                    tokens=len(ids),hit_limit=len(ids)>=512,invalid=pred is None,output=text))
        return records
    try:
        for step in [0,50,100,200,300]:
            adapter=f'{STORE}/outputs/{SFT}' if step==0 else f'{STORE}/outputs/{RUN}' + (f'/checkpoint-{step}' if step<300 else '')
            model=AutoPeftModelForCausalLM.from_pretrained(adapter,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
            greedy=generate(model,rows,1,False)
            result=dict(step=step,adapter=adapter,greedy=greedy,
                greedy_accuracy=sum(r['correct'] for r in greedy)/64,greedy_mean_tokens=sum(r['tokens'] for r in greedy)/64)
            if step in [0,300]:
                samples=[]
                for start in range(0,64,8):
                    set_seed(42+start)
                    samples.extend(generate(model,rows[start:start+8],8,True))
                    print(f'Step {step}: sampled {start+8}/64 questions',flush=True)
                groups=[samples[i:i+8] for i in range(0,len(samples),8)]
                counts=[sum(r['correct'] for r in group) for group in groups]
                result.update(samples=samples,sampled_accuracy=sum(r['correct'] for r in samples)/512,
                    sample_mean_tokens=sum(r['tokens'] for r in samples)/512,
                    pass_at_8=sum(c>0 for c in counts)/64,all_correct_fraction=counts.count(8)/64,
                    all_wrong_fraction=counts.count(0)/64,mixed_fraction=sum(0<c<8 for c in counts)/64,
                    mean_distinct_final_answers=sum(len(set(r['prediction'] for r in g if r['prediction'] is not None)) for g in groups)/64)
            report['checkpoints'].append(result);report['elapsed_seconds']=time.monotonic()-started
            save();print('CHECKPOINT',step,result['greedy_accuracy'],result['greedy_mean_tokens'],flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
        return report
    finally:save()


@app.local_entrypoint()
def main():
    import json
    from datetime import datetime,timezone
    from pathlib import Path
    name=f'grpo-fixed-probe-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}'
    result=probe.remote(name)
    Path('results/grpo300-fixed-probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved results/grpo300-fixed-probe.json; remote results/'+name)
