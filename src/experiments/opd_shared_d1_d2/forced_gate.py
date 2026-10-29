"""Gate-only forced D2 native-cache replay on D1 tokens; never used in training."""
import torch,time
@torch.no_grad()
def forced_logits(model,row,tokenizer,student_depth,eos,forced):
    limit=len(forced);deadline=None;should_stop=None;trace=True
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
                token=torch.tensor([[forced[step]]],device='cuda',dtype=torch.long)
                cache=out.past_key_values;value=int(token.item());answer.append(value)
                del out,logits
                if value==eos:ended=True;break
                x=token;mask=torch.cat([mask,torch.ones_like(token)],dim=1)
        return torch.cat(cached),None if latent is None else latent.detach().cpu()
    finally:
        for module,training in flags:module.training=training
