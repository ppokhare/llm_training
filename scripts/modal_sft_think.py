"""Continue the original SFT adapter for 100 format-adaptation updates."""
from pathlib import Path
import modal
from scripts.modal_sft import image,volume,STORE

app=modal.App('qwen3-gsm8k-sft-think')
SOURCE=f'{STORE}/outputs/sft-20261121-002201-0e7388'


@app.function(image=image,volumes={STORE:volume},cpu=2,memory=8192,
              timeout=180,retries=0,scaledown_window=2)
def check_decoder(run_name: str):
    import hashlib,json
    import torch
    from transformers import AutoTokenizer
    from scripts.data import MODEL_ID
    volume.reload()
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    ids=tok.convert_tokens_to_ids(['<think>','</think>'])
    batch=torch.load(f'{STORE}/outputs/grpo-pilot-20261121-021228-28cae4/failed_microbatch.pt',map_location='cpu',weights_only=True)
    records=[]
    for row in batch['completion_ids']:
        tokens=row.tolist()
        records.append(dict(raw=tok.decode(tokens,skip_special_tokens=False),
            generated_token_ids=tokens,generated_think_token_ids=[t for t in tokens if t in set(ids)]))
    config=json.loads(Path(SOURCE,'adapter_config.json').read_text())
    if config['r']!=16:raise RuntimeError('Source is not rank-16')
    report=dict(source=SOURCE,source_adapter_sha256=hashlib.sha256(Path(SOURCE,'adapter_model.safetensors').read_bytes()).hexdigest(),
        think_token_ids=ids,marked_special=[i in tok.all_special_ids for i in ids],
        decode_skip_true=tok.decode(ids,skip_special_tokens=True),decode_skip_false=tok.decode(ids,skip_special_tokens=False),
        saved_failed_batch=records,source_rank=config['r'])
    output=Path(STORE,'results',f'{run_name}-decoder-audit.json');output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2));volume.commit()
    print(json.dumps(report,indent=2),flush=True)
    return report['source_adapter_sha256']


@app.function(image=image,gpu='H100',volumes={STORE:volume},cpu=2,memory=16384,
              timeout=1200,retries=0,scaledown_window=2)
def continue_sft(run_name: str,expected_source_sha256: str,train_thinking_tokens: bool):
    import hashlib,json,subprocess,sys,time
    volume.reload();started=time.monotonic();output=Path(STORE,'outputs',run_name)
    if output.exists():raise RuntimeError('Output exists')
    def run(command):
        subprocess.run([sys.executable,'-m',*command],cwd='/project',check=True,
                       timeout=max(1,1100-(time.monotonic()-started)))
    try:
        run(['scripts.train_sft','--data',f'{STORE}/data/gsm8k','--sft',SOURCE,
             '--output',str(output),'--thinking-format','--precision','bf16','--attention','flash',
             '--steps','100','--learning-rate','2e-5','--batch-size','32','--gradient-accumulation-steps','1',
             '--eval-batch-size','32','--max-length','512','--eval-at','25,50,100','--eval-on-start',
             '--logging-steps','5','--save-steps','25',
             *(['--train-thinking-tokens'] if train_thinking_tokens else [])])
        if hashlib.sha256(Path(SOURCE,'adapter_model.safetensors').read_bytes()).hexdigest()!=expected_source_sha256:
            raise RuntimeError('Original SFT source weights changed')
        # A fresh process loads the continued adapter onto the original ordinary base.
        run(['scripts.evaluate','--data',f'{STORE}/data/gsm8k','--adapter',str(output),
             '--prompt-style','sft-think','--split','validation','--max-samples','0',
             '--max-new-tokens','512','--batch-size','64','--out',str(output/'validation_sft_think.json')])
        # Check transfer to the separate GRPO prompt on a fixed validation panel.
        run(['scripts.evaluate','--data',f'{STORE}/data/gsm8k','--adapter',str(output),
             '--prompt-style','think','--split','validation','--max-samples','64',
             '--max-new-tokens','512','--batch-size','32','--out',str(output/'validation_grpo_prompt.json')])
        summary=dict(output=str(output),source=SOURCE,source_adapter_sha256=expected_source_sha256,
            source_unchanged=True,elapsed_seconds=time.monotonic()-started)
        (output/'run_summary.json').write_text(json.dumps(summary,indent=2))
        return summary
    finally:volume.commit()


@app.function(image=image,volumes={STORE:volume},cpu=2,memory=8192,
              timeout=180,retries=0,scaledown_window=2)
def inspect_rows():
    """Verify the actual mounted base/head rows on CPU; no optimizer or GPU."""
    import json
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    from scripts.data import MODEL_ID
    volume.reload()
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    ids=tok.convert_tokens_to_ids(['<think>','</think>'])
    model=AutoPeftModelForCausalLM.from_pretrained(SOURCE,is_trainable=True,dtype=torch.bfloat16,device_map='cpu')
    embedding=model.get_input_embeddings();head=model.get_output_embeddings()
    rows=head.weight[ids].detach().float()
    nearby=head.weight[151600:].detach().float()
    matching=(nearby==rows[0]).all(-1).nonzero().flatten()+151600
    report=dict(source=SOURCE,think_token_ids=ids,
        special_token_flags=[i in tok.all_special_ids for i in ids],
        output_head_type=type(head).__name__,head_requires_grad=head.weight.requires_grad,
        embedding_requires_grad=embedding.weight.requires_grad,
        weights_tied=head.weight.data_ptr()==embedding.weight.data_ptr(),
        think_rows_identical=torch.equal(rows[0],rows[1]),row_difference_norm=(rows[0]-rows[1]).norm().item(),
        row_norms=rows.norm(dim=-1).tolist(),identical_reserved_rows=len(matching),
        first_matching_ids=matching[:20].tolist(),
        trainable_names=[n for n,p in model.named_parameters() if p.requires_grad])
    destination=Path(STORE,'results','think-token-row-audit.json')
    destination.write_text(json.dumps(report,indent=2));volume.commit()
    return report


@app.local_entrypoint()
def main(inspect_only: bool = False,train_thinking_tokens: bool = False):
    import json
    from datetime import datetime,timezone
    from uuid import uuid4
    if inspect_only:
        report=inspect_rows.remote()
        Path('results/think-token-row-audit.json').write_text(json.dumps(report,indent=2))
        print({k:v for k,v in report.items() if k!='trainable_names'})
        return
    if not train_thinking_tokens:
        audit=inspect_rows.remote()
        if audit['think_rows_identical'] and not audit['head_requires_grad']:
            raise RuntimeError('Frozen, identical think-token rows cannot learn distinct tags. '
                               'Review the row audit; opt in with --train-thinking-tokens for the selective-row fix.')
    name=f'sft-think-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}'
    print('Format adaptation from original SFT:',name,flush=True)
    source_hash=check_decoder.remote(name)
    result=continue_sft.remote(name,source_hash,train_thinking_tokens)
    Path('results',f'{name}-launch.json').write_text(json.dumps(result,indent=2))
    print(result)
