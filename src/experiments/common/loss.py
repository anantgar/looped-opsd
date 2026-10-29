"""Frozen-teacher full-vocabulary OPSD; reference CE is optional and separate."""
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from experiments.common.rollouts import forward_kl


def backward_opsd(student, teacher_target, rollout, *, depth, denominator,
                 backward, replay_context=None, autocast=nullcontext):
    if denominator < len(rollout.response_lengths):
        raise ValueError('Invalid effective batch denominator')
    if tuple(rollout.prediction_mask.sum(-1).tolist()) != rollout.response_lengths:
        raise ValueError('Response mask/count mismatch')
    context = student.replay_mode(rollout.inputs, replay_context)
    with context:
        with autocast():
            logits = student(rollout.inputs, [depth], rollout.prediction_mask, replay_context)[depth]
            kl = forward_kl(logits, teacher_target.to(logits.device))
            if kl.numel() != sum(rollout.response_lengths):
                raise ValueError('Prediction shift mismatch')
            means = torch.stack([part.mean() for part in kl.split(rollout.response_lengths)])
            loss = means.sum() / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite OPSD')
        stats = {'kl_sum': float(kl.detach().sum()),
                 'kl_sequence_mean_sum': float(means.detach().sum()),
                 'rollout_tokens': kl.numel(), 'rollout_examples': len(means)}
        backward(loss)
    return stats


def backward_reference(student, inputs, targets, *, depth, denominator,
                       weight, backward, autocast=nullcontext):
    mask = targets != -100
    count = int(mask.sum())
    if count == 0 or denominator < count or weight <= 0:
        raise ValueError('Invalid reference CE')
    with autocast():
        logits = student(inputs, [depth], mask, None)[depth]
        value = F.cross_entropy(logits, targets[mask], reduction='sum')
    if not torch.isfinite(value):
        raise FloatingPointError('Nonfinite CE')
    backward(weight * value / denominator)
    return {'reference_ce_sum': float(value.detach()), 'reference_tokens': count}
