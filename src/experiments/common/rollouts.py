"""Ground-truth CE plus forward KL on freshly generated student prefixes.

The caller collects every rollout before any optimizer update, accumulates both
streams across microbatches, then clips/steps AdamW8bit exactly once. No sampling
graph or full-vocabulary target history is stored. Native projection is unchanged.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Callable
import torch
import torch.nn.functional as F

@dataclass
class RolloutBatch:
    inputs: torch.Tensor
    prediction_mask: torch.Tensor
    response_lengths: tuple[int, ...]
    ended: tuple[bool, ...]

def pack_rollouts(prefixes, completions, *, eos, pad, device='cpu'):
    """Predict response tokens from preceding inputs, including genuine EOS.

    Position len(prefix)-1 predicts the FIRST response token. Prompt positions,
    padding and tokens after EOS receive no KD. Cap-truncated rollouts never get
    an invented EOS target. Right padding matches the existing training path.
    """
    if not prefixes or len(prefixes)!=len(completions):
        raise ValueError('Need one nonempty prefix/completion pair per example')
    sequences=[];lengths=[];ended=[]
    for prefix,completion in zip(prefixes,completions):
        if not prefix or not completion:raise ValueError('Empty prefix or rollout')
        if eos in completion[:-1]:raise ValueError('Rollout contains tokens after EOS')
        sequences.append(list(prefix)+list(completion))
        lengths.append(len(completion));ended.append(completion[-1]==eos)
    size=max(len(seq)-1 for seq in sequences)
    inputs=torch.full((len(sequences),size),pad,dtype=torch.long,device=device)
    mask=torch.zeros_like(inputs,dtype=torch.bool)
    for i,(prefix,seq,n) in enumerate(zip(prefixes,sequences,lengths)):
        inputs[i,:len(seq)-1]=torch.tensor(seq[:-1],device=device)
        mask[i,len(prefix)-1:len(prefix)-1+n]=True
    return RolloutBatch(inputs,mask,tuple(lengths),tuple(ended))

def forward_kl(student_logits,teacher_logits):
    if student_logits.shape!=teacher_logits.shape:
        raise ValueError('Teacher and student must use the SAME token prefixes')
    log_teacher=F.log_softmax(teacher_logits.detach().float(),dim=-1)
    log_student=F.log_softmax(student_logits.float(),dim=-1)
    return (log_teacher.exp()*(log_teacher-log_student)).sum(-1)

def backward_microbatch(forward:Callable, gold_inputs, gold_targets,
                        rollout:RolloutBatch, *, maximum_depth,student_depth,
                        kd_weight,reference_token_denominator,
                        rollout_example_denominator,backward:Callable,
                        autocast=nullcontext,replay_context=None):
    """Accumulate CE and KD sequentially, freeing each graph before the next.

    forward(inputs, depths, positions, context) must use native causal exits.
    Reference CE is token-normalized over the effective batch as before. KD is
    token-mean within each completion, then example-mean over the effective
    batch (GKD Eq2/Algorithm1); 512-token failures do not get 512x weight.
    backward is e.g. lambda loss: scaler.scale(loss).backward(). This function
    never zeroes gradients, changes weights, clips or steps the optimizer.
    """
    if not 1<=student_depth<maximum_depth:raise ValueError('Invalid depths')
    if not math.isfinite(kd_weight) or kd_weight<0:raise ValueError('Invalid KD weight')
    positions=gold_targets!=-100
    if positions.sum()==0 or reference_token_denominator<int(positions.sum()):
        raise ValueError('Invalid reference token denominator')
    if rollout_example_denominator<len(rollout.response_lengths):
        raise ValueError('Invalid effective-batch rollout count')
    if tuple(rollout.prediction_mask.sum(-1).tolist())!=rollout.response_lengths:
        raise ValueError('Rollout mask/count mismatch')
    with autocast():
        gold_logits=forward(gold_inputs,[maximum_depth],positions,None)[maximum_depth]
        ce_sum=F.cross_entropy(gold_logits,gold_targets[positions],reduction='sum')
    if not torch.isfinite(ce_sum):raise FloatingPointError('Nonfinite reference CE')
    ce_value=float(ce_sum.detach());backward(ce_sum/reference_token_denominator)
    del gold_logits,ce_sum
    if kd_weight==0:
        return {'reference_ce_sum':ce_value,'reference_tokens':int(positions.sum()),
                'rollout_tokens':sum(rollout.response_lengths),'rollout_examples':len(rollout.response_lengths),'kl_sum':0.,'kl_sequence_mean_sum':0.}
    mode=getattr(forward,'replay_mode',None)
    # Hold replay mode through backward: checkpoint recomputation must use
    # identical dropout/mode settings, not restored reference-training flags.
    with mode(rollout.inputs,replay_context) if mode else nullcontext():
        with autocast():
            logits=forward(rollout.inputs,[student_depth,maximum_depth],rollout.prediction_mask,replay_context)
            student=logits[student_depth];teacher=logits[maximum_depth].detach()
            del logits
            kl=forward_kl(student,teacher)
            if kl.numel()!=sum(rollout.response_lengths):raise ValueError('Wrong prediction shift/count')
            sequence_means=torch.stack([values.mean() for values in kl.split(rollout.response_lengths)])
            kd=kd_weight*sequence_means.sum()/rollout_example_denominator
        if not torch.isfinite(kd):raise FloatingPointError('Nonfinite OPD loss')
        stats={'reference_ce_sum':ce_value,'reference_tokens':int(positions.sum()),
               'rollout_tokens':kl.numel(),'rollout_examples':len(rollout.response_lengths),
               'kl_sum':float(kl.detach().sum()),'kl_sequence_mean_sum':float(sequence_means.detach().sum())}
        backward(kd)
    return stats
