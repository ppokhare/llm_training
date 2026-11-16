"""Small PyTorch utilities for decoding, on-policy rollouts, RL, and distillation.

Uses only torch. All log-probabilities are natural logarithms.
Assumes unpadded, equal-length prompts when batching completion scoring.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0  # 0 means greedy
    top_k: int | None = None  # None disables top-k
    top_p: float = 1.0        # 1.0 disables nucleus filtering


def _validate_sampling(config: SamplingConfig) -> None:
    if not math.isfinite(config.temperature) or config.temperature < 0:
        raise ValueError("temperature must be a finite number >= 0")
    if config.top_k is not None and (not isinstance(config.top_k, int) or config.top_k < 1):
        raise ValueError("top_k must be None or an integer >= 1")
    if not math.isfinite(config.top_p) or not (0 < config.top_p <= 1):
        raise ValueError("top_p must be in (0, 1]")


def sampling_logits(raw_logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    """Warp logits in order: temperature -> top-k -> top-p.

    Input/output: [batch_size, vocab_size]. -inf marks excluded tokens.
    Does not perform sampling. Uses float32 to make softmax/log-softmax stable.
    """
    _validate_sampling(config)
    if raw_logits.ndim != 2:
        raise ValueError("raw_logits must have shape [batch_size, vocab_size]")

    logits = raw_logits.float()
    if config.temperature == 0:
        # Deterministic policy; ignored top-k and top-p.
        best = logits.argmax(dim=-1, keepdim=True)
        return torch.full_like(logits, -torch.inf).scatter(-1, best, 0.0)

    logits = logits / config.temperature

    if config.top_k is not None and config.top_k < logits.size(-1):
        # scatter gives exactly k eligible tokens, even when logits tie.
        values, indices = torch.topk(logits, k=config.top_k, dim=-1)
        logits = torch.full_like(logits, -torch.inf).scatter(-1, indices, values)

    if config.top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs > config.top_p
        # Keep the first token that makes cumulative probability reach top_p.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        logits = torch.empty_like(logits).scatter(-1, sorted_indices, sorted_logits)

    return logits


def categorical_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Entropy in nats, with safe handling of log(0) = -inf."""
    probs = log_probs.exp()
    safe_log_probs = torch.where(probs > 0, log_probs, torch.zeros_like(log_probs))
    return -(probs * safe_log_probs).sum(dim=-1)


@torch.no_grad()
def sample_next_token(raw_logits: torch.Tensor, config: SamplingConfig) -> dict[str, torch.Tensor]:
    """Sample next token and retain log p under TWO policies.

    model_logprob = log p_model(token | prefix), before sampling warpers.
    behavior_logprob = log p_sampling(token | prefix), after temperature/top-k/top-p.
    With (temperature=1, top_k=None, top_p=1), these match exactly.
    """
    model_log_probs = F.log_softmax(raw_logits.float(), dim=-1)
    behavior_log_probs = F.log_softmax(sampling_logits(raw_logits, config), dim=-1)
    token_ids = torch.multinomial(behavior_log_probs.exp(), num_samples=1)
    return {
        "token_ids": token_ids,  # [B, 1]
        "model_logprob": model_log_probs.gather(-1, token_ids),  # [B, 1]
        "behavior_logprob": behavior_log_probs.gather(-1, token_ids),  # [B, 1]
        "behavior_entropy": categorical_entropy(behavior_log_probs),  # [B]
    }


@torch.no_grad()
def rollout_with_logprobs(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    config: SamplingConfig,
    eos_token_id: int | None = None,
) -> dict[str, torch.Tensor]:
    """KV-cached single-example rollout; includes EOS if sampled.

    Returns completion_ids, model_logprobs, behavior_logprobs [1, T].
    Saving logprobs here is for tracking the behavior/old policy; to train,
    re-run completion_logprobs WITHOUT no_grad to obtain fresh gradients.
    """
    if prompt_ids.ndim != 2 or prompt_ids.size(0) != 1 or prompt_ids.size(1) == 0:
        raise ValueError("prompt_ids must have shape [1, prompt_length>=1]")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    model.eval()
    outputs = model(prompt_ids, use_cache=True)
    cache = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]

    tokens, model_lps, behavior_lps, entropies = [], [], [], []
    for step in range(max_new_tokens):
        sample = sample_next_token(next_logits, config)
        next_token = sample["token_ids"]
        tokens.append(next_token)
        model_lps.append(sample["model_logprob"])
        behavior_lps.append(sample["behavior_logprob"])
        entropies.append(sample["behavior_entropy"].unsqueeze(1))

        if eos_token_id is not None and bool((next_token == eos_token_id).all().item()):
            break
        if step < max_new_tokens - 1:
            outputs = model(next_token, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

    return {
        "completion_ids": torch.cat(tokens, dim=1),
        "model_logprobs": torch.cat(model_lps, dim=1),
        "behavior_logprobs": torch.cat(behavior_lps, dim=1),
        "behavior_entropy": torch.cat(entropies, dim=1),
    }


def completion_logprobs(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
) -> torch.Tensor:
    """Teacher-forced, differentiable model log p for completion tokens.

    Prompt: [B, P], completion: [B, C]. Returns [B, C].
    Shift alignment: logits at position P-1 predict the FIRST completion token.
    No padding or variable-length prompts in this minimal implementation.
    """
    if (prompt_ids.ndim != 2 or completion_ids.ndim != 2 or
        prompt_ids.size(0) != completion_ids.size(0) or
        prompt_ids.size(1) == 0 or completion_ids.size(1) == 0):
        raise ValueError("Use [B, P>=1] prompts and [B, C>=1] completions")

    input_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
    logits = model(input_ids, use_cache=False).logits
    start = prompt_ids.size(1) - 1
    end = start + completion_ids.size(1)
    completion_logits = logits[:, start:end, :].float()
    return F.log_softmax(completion_logits, dim=-1).gather(
        -1, completion_ids.unsqueeze(-1)
    ).squeeze(-1)


def sequence_logprob(token_logprobs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Sum of chosen-token log probabilities, per example [B]."""
    if mask is None:
        return token_logprobs.sum(dim=-1)
    return (token_logprobs * mask).sum(dim=-1)


def tokenwise_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    direction: str = "forward",
) -> torch.Tensor:
    """Per-position KL over the entire vocab; shape [B, T].

    forward: KL(teacher || student), akin to soft-label cross-entropy.
    reverse: KL(student || teacher), at the SAME fixed prefixes.
    Teacher is detached, student retains gradient.
    """
    student_lp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_lp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    if direction == "forward":
        return (teacher_lp.exp() * (teacher_lp - student_lp)).sum(dim=-1)
    if direction == "reverse":
        return (student_lp.exp() * (student_lp - teacher_lp)).sum(dim=-1)
    raise ValueError("direction must be 'forward' or 'reverse'")


def masked_mean(losses: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Average over valid tokens, or over all entries when mask is absent."""
    if mask is None:
        return losses.mean()
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def reinforce_loss(
    new_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sequence-level REINFORCE surrogate, with stop-grad advantages [B]."""
    if advantages.ndim != 1:
        raise ValueError("advantages must have shape [B]")
    return -(sequence_logprob(new_logprobs, mask) * advantages.detach()).mean()


def ppo_clipped_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor | None = None,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Minimal tokenwise PPO policy surrogate (no value or reference KL loss).

    new/old: [B,T], advantages: [B] or [B,T], mask: optional [B,T].
    This assumes old/new logprobs refer to the SAME policy definition.
    """
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(1)
    ratio = (new_logprobs - old_logprobs.detach()).exp()
    advantage = advantages.detach()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage
    return masked_mean(-torch.minimum(unclipped, clipped), mask)
