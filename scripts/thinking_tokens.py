"""Audit frozen tag rows and selectively extend an existing LoRA adapter."""
import torch
from peft import get_peft_model_state_dict
from peft.utils.other import set_additional_trainable_modules


def audit_thinking_head(model, token_ids):
    head=model.get_output_embeddings()
    rows=head.weight[token_ids].detach()
    return dict(token_ids=list(token_ids),rows_identical=torch.equal(rows[0],rows[1]),
                output_head_trainable=any(p.requires_grad for p in head.parameters()),
                tied_to_input=head.weight.data_ptr()==model.get_input_embeddings().weight.data_ptr())


def enable_thinking_token_training(model, token_ids):
    """Keep existing LoRA weights/rank, add only the selected tied token rows."""
    if len(model.peft_config)!=1:raise ValueError('Expected one existing LoRA adapter')
    adapter=next(iter(model.peft_config));config=model.peft_config[adapter]
    if config.r!=16:raise ValueError('Expected original rank-16 adapter')
    before={n:t.detach().clone() for n,t in get_peft_model_state_dict(model).items() if 'lora_' in n}
    config.trainable_token_indices=list(token_ids)
    config.ensure_weight_tying=True
    # Native PEFT auxiliary modules: no new LoRA adapter or LoRA reinjection.
    set_additional_trainable_modules(model.base_model,config,model.config.to_dict(),adapter)
    after=get_peft_model_state_dict(model)
    if any(not torch.equal(t,after[n]) for n,t in before.items()):
        raise RuntimeError('Adding token rows altered the existing LoRA state')
    rows=[]
    for name,p in model.named_parameters():
        if 'trainable_tokens_delta' in name:
            p.data=p.data.float()
            rows.append(p)
    count=sum(p.numel() for p in rows)
    expected=len(token_ids)*model.get_input_embeddings().weight.shape[1]
    if count!=expected:raise RuntimeError(f'Expected {expected} token parameters, got {count}')
    return dict(token_ids=list(token_ids),additional_parameters=count,adapter_count=len(model.peft_config),
                lora_rank=config.r,existing_lora_unchanged=True,ensure_weight_tying=True)
