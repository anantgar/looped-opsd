"""Dense reverse KL for shallow OPSD; retain forward KL for the D4 anchor."""

from contextlib import nullcontext
import time
import torch
import torch.nn.functional as F
from experiments.common.rollouts import forward_kl
from experiments.common.loss import backward_reference


def reverse_kl(student_logits, teacher_logits):
    """KL(student || frozen teacher), one value per response position.

    Differentiate through BOTH student probabilities and log probabilities.
    This is exact vocabulary summation at the sampled response prefixes, not
    the sampled-token policy-gradient estimator used by LoOPSD.
    """
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    log_student = F.log_softmax(student_logits.to(dtype), dim=-1)
    log_teacher = F.log_softmax(teacher_logits.detach().to(dtype), dim=-1)
    return (log_student.exp() * (log_student - log_teacher)).sum(dim=-1)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def backward_opsd(
    student,
    teacher_target,
    rollout,
    *,
    depth,
    denominator,
    backward,
    replay_context=None,
    autocast=nullcontext
):
    if denominator < len(rollout.response_lengths):
        raise ValueError("Invalid effective batch denominator")
    if tuple(rollout.prediction_mask.sum(-1).tolist()) != rollout.response_lengths:
        raise ValueError("Response mask/count mismatch")
    with student.replay_mode(rollout.inputs, replay_context):
        _sync()
        start = time.perf_counter()
        with autocast():
            logits = student(
                rollout.inputs, [depth], rollout.prediction_mask, replay_context
            )[depth]
            target = teacher_target.to(logits.device)
            # The anchor is deliberately unchanged: reverse only the shallow loss.
            values = (
                forward_kl(logits, target) if depth == 4 else reverse_kl(logits, target)
            )
            if values.numel() != sum(rollout.response_lengths):
                raise ValueError("Prediction shift mismatch")
            means = torch.stack(
                [part.mean() for part in values.split(rollout.response_lengths)]
            )
            loss = means.sum() / denominator
        _sync()
        forward_seconds = time.perf_counter() - start
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite distillation loss")
        stats = {
            "kl_sum": float(values.detach().sum()),
            "kl_sequence_mean_sum": float(means.detach().sum()),
            "rollout_tokens": values.numel(),
            "rollout_examples": len(means),
            "student_forward_seconds": forward_seconds,
        }
        start = time.perf_counter()
        backward(loss)
        _sync()
        stats["backward_seconds"] = time.perf_counter() - start
    return stats
