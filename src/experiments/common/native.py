"""Reuse the pinned native cache/generation and full-projection training paths.

Configure PYTHONPATH to the immutable matched-source/src and its pinned
base_depth_eval dependency. Never substitute shallow KV into a deep cache.
CUDA cached/replay-logit and gradient/memory gates are mandatory before launch.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import torch
import time

@dataclass
class NativeRollout:
    prefix: list[int]
    completion: list[int]
    initial_state: object
    ended: bool
    student_depth: int
    cached_logits: object = None

@torch.no_grad()
def collect_greedy_rollout(model,row,tokenizer,student_depth,eos,*,deadline=None,should_stop=None):
    from native_recipe_eval import generate,evaluation_weights,compat_tiny
    from base_depth_eval import states
    prefix=tokenizer.encode(row['prompt'],add_special_tokens=False)
    flags=[(m,m.training) for m in model.modules()]
    model.eval()
    try:
        if model.kind=='tinyllama':compat_tiny(model.base)
        latent=states(model.base,model.kind,row,len(prefix)+512)
        with evaluation_weights(model.base):
            answers,ended,_=generate(model.base,model.kind,[prefix],student_depth,latent,eos,512,deadline,should_stop)
        return NativeRollout(prefix,answers[0],None if latent is None else latent.detach().cpu(),ended[0],student_depth)
    finally:
        for module,training in flags:module.training=training

@torch.no_grad()
def collect_sampled_rollout(model,row,tokenizer,student_depth,eos,*,generator=None,deadline=None,should_stop=None,limit=512,trace=False):
    """Categorical T=1 rollout, using the same native cache APIs as evaluation.

    Evaluation stays greedy. Persist this generator's state if a separate
    generator is supplied; otherwise the existing CUDA RNG checkpoint applies.
    This samples ONLY student logits; teacher targets are recomputed on replay.
    """
    from native_recipe_eval import evaluation_weights,compat_tiny
    from base_depth_eval import states,tiny_cache,DETAILS
    if not 1<=limit<=512:raise ValueError('Invalid rollout limit')
    prefix=tokenizer.encode(row['prompt'],add_special_tokens=False)
    flags=[(m,m.training) for m in model.modules()];model.eval()
    try:
        base=model.base;size=len(prefix)
        x=torch.tensor([prefix],device='cuda',dtype=torch.long)
        mask=torch.ones_like(x);latent=states(base,model.kind,row,size+512)
        if model.kind=='tinyllama':
            compat_tiny(base);cache=tiny_cache(base,student_depth,1,size+512)
        else:
            from transformers.cache_utils import DynamicCache
            cache=DynamicCache();cache.append_new_layers(student_depth*base.config.num_hidden_layers-1)
            base.model.total_ut_steps=student_depth;base.config.total_ut_steps=student_depth;base.model.config.total_ut_steps=student_depth
        answer=[];ended=False;cached=[]
        with evaluation_weights(base):
            for step in range(limit):
                if should_stop is not None and should_stop():raise InterruptedError('Rollout interrupted')
                if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Rollout deadline')
                if model.kind=='tinyllama':
                    start=0 if step==0 else size+step-1
                    out=base(x,input_states=latent[:,start:start+x.shape[1]],num_steps=[0,student_depth],past_key_values=cache,use_cache=True,cache_position=torch.arange(start,start+x.shape[1],device='cuda'),output_details=DETAILS)
                else:
                    pos=mask.cumsum(-1)-1
                    if step:pos=pos[:,-1:]
                    out=base(x,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,exit_at_step=student_depth-1,use_weighted_exit=False,logits_to_keep=1)
                logits=out.logits[:,-1].float()
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite rollout logits')
                if trace:cached.append(logits.detach().cpu())
                token=torch.multinomial(torch.softmax(logits,dim=-1),1,generator=generator)
                cache=out.past_key_values;value=int(token.item());answer.append(value)
                del out,logits
                if value==eos:ended=True;break
                x=token;mask=torch.cat([mask,torch.ones_like(token)],dim=1)
        return NativeRollout(prefix,answer,None if latent is None else latent.detach().cpu(),ended,student_depth,torch.cat(cached) if trace else None)
    finally:
        for module,training in flags:module.training=training

class NativeForward:
    def __init__(self,model):self.model=model
    @contextmanager
    def _replay(self,tokens,context):
        m=self.model;flags=[(module,module.training) for module in m.modules()]
        original=m.base.forward
        try:
            # Keep wrapper checkpointing enabled while disabling base dropout,
            # so replay matches the no-grad generation policy's forward map.
            m.train();m.base.eval()
            if m.kind=='tinyllama':
                if context is None:raise ValueError('Tiny rollout must replay identical native initial states')
                latent=context.to(tokens.device)[:,:tokens.shape[1]]
                if latent.shape[:2]!=tokens.shape:raise ValueError('Initial-state shape mismatch')
                def fixed(*args,**kwargs):
                    kwargs['input_states']=latent
                    return original(*args,**kwargs)
                m.base.forward=fixed
            elif context is not None:raise ValueError('Ouro does not use Tiny initial states')
            yield
        finally:
            m.base.forward=original
            for module,training in flags:module.training=training
    def __call__(self,inputs,depths,positions,context):
        return self.model(inputs,depths,positions)
    def replay_mode(self,inputs,context):
        return self._replay(inputs,context)
