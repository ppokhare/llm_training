"""Selective tag rows must break tied-logit symmetry and survive PEFT reload."""
import tempfile
import unittest
import torch
from peft import get_peft_model,LoraConfig,PeftModel
from transformers import Qwen3Config,Qwen3ForCausalLM
from scripts.thinking_tokens import audit_thinking_head,enable_thinking_token_training


class ThinkingTokens(unittest.TestCase):
    def test_selected_rows_preserve_lora_and_round_trip(self):
        torch.manual_seed(42)
        cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,tie_word_embeddings=True)
        base=Qwen3ForCausalLM(cfg)
        with torch.no_grad():base.get_input_embeddings().weight[31].copy_(base.get_input_embeddings().weight[30])
        original={n:t.detach().clone() for n,t in base.state_dict().items()}
        model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,target_modules=['q_proj'],task_type='CAUSAL_LM'))
        self.assertTrue(audit_thinking_head(model,[30,31])['rows_identical'])
        info=enable_thinking_token_training(model,[30,31])
        self.assertEqual(info['additional_parameters'],32)
        self.assertEqual(info['adapter_count'],1)
        self.assertEqual(model.peft_config['default'].r,16)
        hidden=torch.randn(1,16)
        logits=model.get_output_embeddings()(hidden)
        torch.nn.functional.cross_entropy(logits,torch.tensor([30])).backward()
        torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1).step()
        self.assertFalse(torch.equal(model.get_output_embeddings().weight[30],model.get_output_embeddings().weight[31]))
        self.assertTrue(torch.equal(model.get_input_embeddings().weight,model.get_output_embeddings().weight))
        self.assertTrue(torch.equal(model.get_input_embeddings().weight[:30],original['model.embed_tokens.weight'][:30]))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            fresh=Qwen3ForCausalLM(cfg);fresh.load_state_dict(original)
            loaded=PeftModel.from_pretrained(fresh,directory)
            self.assertTrue(torch.equal(model.get_output_embeddings().weight,loaded.get_output_embeddings().weight))
            self.assertEqual(len(loaded.peft_config),1)


class FreshThinkingTokens(unittest.TestCase):
    def test_fused_completion_loss_reaches_bf16_tied_rows(self):
        from trl.trainer.utils import add_fused_lm_head
        from scripts.sft_thinking_checks import FreshThinkingChecks
        from scripts.train_sft import fp8_layer_filter

        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=8, tie_word_embeddings=True)
        base = Qwen3ForCausalLM(config).to(torch.bfloat16)
        with torch.no_grad():
            base.get_input_embeddings().weight[31].copy_(base.get_input_embeddings().weight[30])
        add_fused_lm_head(base)
        model = get_peft_model(base, LoraConfig(
            r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM",
            trainable_token_indices=[30, 31], ensure_weight_tying=True))
        checks = FreshThinkingChecks(model, [30, 31], ".")
        eligible = [name for name, module in model.named_modules() if fp8_layer_filter(module, name)]
        self.assertTrue(eligible)
        self.assertFalse(any("lm_head" in name for name in eligible))
        output = model(input_ids=torch.tensor([[1, 2, 30, 3, 31]]),
                       labels=torch.tensor([[-100, -100, 30, 3, 31]]), fused_lm_head=True)
        self.assertEqual(output.label_mask.sum().item(), 3)
        (-output.log_probs.sum()).backward()
        gradient = checks.rows.grad
        self.assertEqual(gradient.dtype, torch.bfloat16)
        self.assertTrue((gradient.float().norm(dim=1) > 0).all())
        self.assertFalse(torch.equal(gradient[0], gradient[1]))


class ThinkingFp8Trace(unittest.TestCase):
    def test_requires_cuda_correlation_without_requiring_an_fp8_name(self):
        import json
        from pathlib import Path
        from scripts.train_sft import audit_fp8_cuda_trace

        dispatch = {"cat": "cpu_op", "name": "aten::_scaled_mm", "args": {"External id": 17}}
        kernel = {"cat": "kernel", "name": "nvjet_sm90_qrtst_TNT", "args": {"External id": 17}}
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory, "trace.json")
            trace.write_text(json.dumps({"traceEvents": [dispatch, kernel]}))
            self.assertEqual(audit_fp8_cuda_trace(trace)["linked_dispatches"], 1)
            kernel["args"]["External id"] = 18
            trace.write_text(json.dumps({"traceEvents": [dispatch, kernel]}))
            with self.assertRaisesRegex(RuntimeError, "0/1 linked"):
                audit_fp8_cuda_trace(trace)


if __name__ == '__main__':
    unittest.main()
