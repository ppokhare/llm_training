"""Exact-step loss evaluation and raw generated-format checks for SFT continuation."""
import json
from pathlib import Path
import time
import torch
from transformers import TrainerCallback
from scripts.evaluate import make_record


class EvaluateAt(TrainerCallback):
    """Request evaluations/checkpoints only at the explicitly selected update counts."""
    def __init__(self, steps): self.steps=set(steps)

    def on_step_end(self,args,state,control,**kwargs):
        control.should_evaluate=state.global_step in self.steps
        control.should_save=state.global_step in self.steps
        return control


class ThinkingGenerationChecks(TrainerCallback):
    """Keep the same validation panel and preserve raw IDs/text at every evaluation."""
    def __init__(self,rows,tokenizer,output,batch_size=32):
        self.rows=rows;self.tokenizer=tokenizer;self.output=Path(output);self.batch_size=batch_size
        self.history=[]

    def on_evaluate(self,args,state,control,model=None,metrics=None,**kwargs):
        tokenizer=self.tokenizer;old_padding=tokenizer.padding_side;was_training=model.training
        tokenizer.padding_side='left';model.eval();records=[]
        eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
        try:
            for start in range(0,len(self.rows),self.batch_size):
                rows=self.rows[start:start+self.batch_size]
                inputs=tokenizer([r['prompt'] for r in rows],padding=True,return_tensors='pt').to(next(model.parameters()).device)
                began=time.monotonic()
                with torch.inference_mode():
                    output=model.generate(**inputs,max_new_tokens=512,do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id)
                for i,row in enumerate(rows):
                    ids=output[i,inputs['input_ids'].shape[1]:].tolist()
                    end=next((j+1 for j,t in enumerate(ids) if t in eos),len(ids));ids=ids[:end]
                    records.append(make_record(row,ids,tokenizer,eos,512,int(inputs['attention_mask'][i].sum()),time.monotonic()-began))
        finally:
            tokenizer.padding_side=old_padding
            model.train(was_training)
        n=len(records)
        summary=dict(step=state.global_step,n=n,accuracy=sum(r['correct'] for r in records)/n,
            think_format_rate=sum(r['think_format_valid'] for r in records)/n,
            opening_tag_count=sum(r['think_token_ids'][0] in r['generated_think_token_ids'] for r in records),
            closing_tag_count=sum(r['think_token_ids'][1] in r['generated_think_token_ids'] for r in records),
            starts_with_opening_tag_count=sum(r['generated_token_ids'][:1] == r['think_token_ids'][:1] for r in records),
            valid_final_count=sum(r['prediction'] is not None and r['final_tag_count'] == 1 for r in records),
            mean_output_tokens=sum(r['generated_tokens'] for r in records)/n,
            token_limit_hits=sum(r['hit_token_limit'] for r in records),teacher_forced=metrics or {})
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/f'generated_step_{state.global_step}.json').write_text(json.dumps(dict(summary=summary,samples=records),indent=2))
        self.history.append(summary)
        (self.output/'generated_history.json').write_text(json.dumps(self.history,indent=2))
        print('FIXED GENERATED CHECK',json.dumps(summary),flush=True)
        if state.global_step==0:
            print('RAW STEP-0 EXAMPLE:',records[0]['output_with_special_tokens'],flush=True)
            print('THINK TOKEN IDS:',records[0]['think_token_ids'],'GENERATED THINK TOKENS:',records[0]['generated_think_token_ids'],flush=True)


class FreshThinkingChecks(TrainerCallback):
    """Require separate tied-row gradients, output logits and persisted updates."""

    def __init__(self, model, ids, output):
        from peft import get_peft_model_state_dict

        self.model = model
        self.ids = ids
        self.output = Path(output)
        self.initial = {name: value.detach().cpu().clone()
                        for name, value in get_peft_model_state_dict(model, save_embedding_layers=False).items()}
        trainable = {name: parameter for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
        rows = [parameter for name, parameter in trainable.items() if 'trainable_tokens_delta' in name]
        shape = (2, model.config.hidden_size)
        if len(rows) != 1 or tuple(rows[0].shape) != shape or not model.config.tie_word_embeddings:
            raise RuntimeError('Expected one shared PEFT parameter for the two tied rows')
        if any('lora_' not in name and 'trainable_tokens_delta' not in name for name in trainable):
            raise RuntimeError('Unexpected trainable base weights')
        for parameter in trainable.values():
            parameter.data = parameter.data.to(torch.bfloat16)
        self.rows = rows[0]
        self.report = {'token_ids': ids, 'trainable_dtype': 'torch.bfloat16',
                       'token_parameter_shape': list(shape), 'trainable_tensors': len(trainable)}
        self.check_tying()
        # TRL's fused loss reads weight and bias separately. PEFT exposes only weight.
        head = model.get_output_embeddings()
        if head.original_module.bias is not None:
            raise RuntimeError("Expected a bias-free Qwen output head")
        head.bias = None

    def check_tying(self):
        embedding = self.model.get_input_embeddings()
        head = self.model.get_output_embeddings()
        for module in (embedding, head):
            if module.original_module.weight.requires_grad:
                raise RuntimeError('Expected frozen original embedding and head weights')
        if embedding.original_module.weight is not head.original_module.weight:
            raise RuntimeError('Original embedding and output weights are no longer tied')
        if not torch.equal(embedding.weight[self.ids], head.weight[self.ids]):
            raise RuntimeError('Effective input and output rows differ')

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step:
            return
        gradient = self.rows.grad
        if gradient is None or self.rows.dtype != torch.bfloat16:
            raise RuntimeError('Missing BF16 token-row gradients')
        gradient = gradient.float()
        norms = gradient.norm(dim=1)
        difference = (gradient[0] - gradient[1]).norm().item()
        if not torch.isfinite(gradient).all() or not (norms > 0).all() or not difference:
            raise RuntimeError('Thinking token rows lack distinct nonzero gradients')
        self.report.update(gradient_norms=norms.tolist(), gradient_difference_norm=difference)

    def verify_checkpoint(self):
        from peft import get_peft_model_state_dict, load_peft_weights, PeftModel
        from transformers import AutoModelForCausalLM

        self.check_tying()
        current = get_peft_model_state_dict(self.model, save_embedding_layers=False)
        changed = [name for name, value in current.items()
                   if not torch.equal(self.initial[name], value.detach().cpu())]
        row_keys = [name for name in changed if 'trainable_tokens_delta' in name]
        if len(row_keys) != 1 or not any('lora_' in name for name in changed):
            raise RuntimeError('Expected both LoRA and token-row updates')
        name = row_keys[0]
        if not (current[name].detach().cpu() != self.initial[name]).any(dim=1).all():
            raise RuntimeError('Expected both selected token rows to update')
        saved = load_peft_weights(str(self.output), device='cpu')
        if any(name not in saved or not torch.equal(value.detach().cpu(), saved[name])
               for name, value in current.items()):
            raise RuntimeError('Adapter tensors differ from the saved checkpoint')
        base = AutoModelForCausalLM.from_pretrained(
            self.model.config._name_or_path, revision=getattr(self.model.config, '_commit_hash', None),
            dtype=torch.bfloat16)
        loaded = PeftModel.from_pretrained(base, self.output, autocast_adapter_dtype=False).eval()
        restored = get_peft_model_state_dict(loaded, save_embedding_layers=False)
        if any(not torch.equal(value.detach().cpu(), restored[name]) for name, value in current.items()):
            raise RuntimeError('Reloaded adapter tensors differ from live tensors')
        live_rows = self.model.get_output_embeddings().weight[self.ids].detach().cpu()
        if not torch.equal(loaded.get_input_embeddings().weight[self.ids], live_rows):
            raise RuntimeError('Reloaded input rows differ from trained rows')
        direction = (live_rows[0].float() - live_rows[1].float()).to(torch.bfloat16)
        with torch.no_grad():
            logits = loaded.get_output_embeddings()(direction.unsqueeze(0))[0, self.ids]
        gap = (logits[0].float() - logits[1].float()).item()
        if gap <= 0:
            raise RuntimeError('Could not separate effective output logits for the thinking tags')
        self.report.update(changed_adapter_tensors=len(changed), reload_matches=True,
                           output_logits=logits.tolist(), logit_gap=gap)
        (self.output / 'thinking_token_checks.json').write_text(json.dumps(self.report, indent=2))


def audit_thinking_labels(model, tokenizer, batch, token_ids):
    """Score the next-token targets at the actual collator's unmasked positions."""
    device = next(model.parameters()).device
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention = batch["attention_mask"].to(device)
    positions = []
    for ids, targets, mask in zip(inputs, labels, attention):
        unmasked = targets.ne(-100).nonzero().flatten()
        indices = [targets.eq(token).nonzero().flatten().tolist() for token in token_ids]
        if any(len(index) != 1 for index in indices):
            raise RuntimeError("Expected one supervised opening and one supervised closing tag")
        opening, closing = (index[0] for index in indices)
        tail = tokenizer.decode(targets[closing + 1:][targets[closing + 1:].ne(-100)].tolist())
        if (opening == 0 or opening != unmasked[0].item() or closing <= opening + 1
                or not torch.equal(ids[unmasked], targets[unmasked])
                or not mask[unmasked].all() or not tail.lstrip().startswith("<final>")):
            raise RuntimeError("Thinking supervision positions or completion mask are incorrect")
        positions.append([opening, closing])

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            base = model.get_base_model()
            hidden = base.model(input_ids=inputs, attention_mask=attention,
                                use_cache=False).last_hidden_state
            indices = torch.tensor(positions, device=device) - 1
            predictors = hidden[torch.arange(len(positions), device=device)[:, None], indices]
            logits = model.get_output_embeddings()(predictors).float()
            probability = logits.softmax(dim=-1)
    finally:
        model.train(was_training)
    examples = []
    for i, pair in enumerate(positions):
        measurements = {}
        for j, name in enumerate(("opening", "closing")):
            scores = logits[i, j]
            target, other = token_ids[j], token_ids[1 - j]
            best_other = scores.clone()
            best_other[target] = -torch.inf
            measurements[name] = {
                "label_position": pair[j], "predictor_position": pair[j] - 1,
                "rank": int((scores > scores[target]).sum()) + 1,
                "probability": probability[i, j, target].item(),
                "margin_vs_top_competitor": (scores[target] - best_other.max()).item(),
                "margin_vs_other_tag": (scores[target] - scores[other]).item(),
                "top_token_id": scores.argmax().item(),
            }
        examples.append(measurements)
    return examples


class ThinkingResumeChecks(TrainerCallback):
    """Audit the real first resumed batch and reuse it at evaluation milestones."""

    def __init__(self, tokenizer, token_ids, output, generation, pause_at_update=None):
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.output = Path(output)
        self.generation = generation
        self.pause_at_update = pause_at_update
        self.batch = None

    def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, model=None, **kwargs):
        rows = next(p for name, p in model.named_parameters()
                    if p.requires_grad and "trainable_tokens_delta" in name)
        optimizer_state = optimizer.state[rows]
        trainable_dtypes = {str(p.dtype) for p in model.parameters() if p.requires_grad}
        if trainable_dtypes != {"torch.bfloat16"}:
            raise RuntimeError(f"Resume changed the trainable parameter dtype: {trainable_dtypes}")
        lr_scheduler = getattr(lr_scheduler, "scheduler", lr_scheduler)
        if int(optimizer_state["step"]) != state.global_step or lr_scheduler.last_epoch != state.global_step:
            raise RuntimeError("Optimizer token-row moments or scheduler position did not restore")
        previous = [group["lr"] for group in optimizer.param_groups]
        saved_bases = list(lr_scheduler.base_lrs)
        if self.pause_at_update is not None:
            lr_scheduler.base_lrs = [args.learning_rate for _ in saved_bases]
            for group, base in zip(optimizer.param_groups, lr_scheduler.base_lrs):
                group["initial_lr"] = base
        # Keep moments and step count. Extend the exhausted 10-step linear schedule.
        current = [base * schedule(lr_scheduler.last_epoch)
                   for base, schedule in zip(lr_scheduler.base_lrs, lr_scheduler.lr_lambdas)]
        for group, rate in zip(optimizer.param_groups, current):
            group["lr"] = rate
        lr_scheduler._last_lr = current
        (self.output / "resume_optimizer.json").write_text(json.dumps({
            "trainable_dtype": "torch.bfloat16", "restored_step": state.global_step, "token_row_adam_step": int(optimizer_state["step"]),
            "restored_parameter_states": len(optimizer.state),
            "checkpoint_learning_rates": previous, "extended_learning_rates": current,
            "schedule_end_step": state.max_steps, "checkpoint_base_lrs": saved_bases,
            "extended_base_lrs": lr_scheduler.base_lrs,
        }, indent=2))

    def first_training_batch(self, model, batch, state):
        if self.batch is not None:
            return
        self.batch = {name: batch[name].detach().clone() for name in ("input_ids", "labels", "attention_mask")}
        self.record(model, state.global_step)
        print(f"Verified both unmasked thinking labels in {len(self.batch['labels'])} training examples", flush=True)

    def record(self, model, step, optimizer=None):
        records = audit_thinking_labels(model, self.tokenizer, self.batch, self.token_ids)
        metrics = {kind: {key: sum(r[kind][key] for r in records) / len(records)
                          for key in ("rank", "probability", "margin_vs_top_competitor", "margin_vs_other_tag")}
                   for kind in ("opening", "closing")}
        payload = {"step": step, "examples": records, "means": metrics}
        if optimizer is not None:
            payload["learning_rates"] = [group["lr"] for group in optimizer.param_groups]
        (self.output / f"thinking_logits_step_{step}.json").write_text(json.dumps(payload, indent=2))
        print("THINKING LOGITS", json.dumps({"step": step, **metrics}), flush=True)

    def on_evaluate(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self.batch is not None:
            self.record(model, state.global_step, optimizer)
        if self.pause_at_update == state.global_step:
            control.should_training_stop = True
        if state.global_step == 50 and self.generation is not None:
            improved = any(row["think_format_rate"] > 0 for row in self.generation.history)
            print(f"Step 50 decision: complete thinking blocks improved over step 10: {improved}", flush=True)
            if not improved:
                control.should_training_stop = True
        return control
