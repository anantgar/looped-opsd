"""Matched native generation: unchanged prompts, targets, EOS and 512-token cap.
Fixed latent states, identical groups across checkpoints; shape-counted FLOPs.
"""
import contextlib, json, time, sys
import torch
from answer_extraction import grade
from base_depth_eval import states, tiny_cache, groups_for, DETAILS, raven

class ComputeCounter:
    """Executed linear and SDPA matmul FLOPs; multiply+add counts as2.
    Excludes norms, softmax, activations and memory movement. Attention counts
    dense padded matrices, even where FlashAttention may skip masked tiles.
    """
    def __init__(self,base):self.base=base;self.linear=0;self.attention=0;self.handles=[]
    def __enter__(self):
        def linear(layer,args,out):self.linear+=2*args[0].numel()*layer.out_features
        for layer in self.base.modules():
            if isinstance(layer,torch.nn.Linear):self.handles.append(layer.register_forward_hook(linear))
        self.original=torch.nn.functional.scaled_dot_product_attention
        def attention(q,k,v,*a,**kw):
            self.attention+=4*q.shape[0]*q.shape[1]*q.shape[-2]*k.shape[-2]*q.shape[-1]
            return self.original(q,k,v,*a,**kw)
        torch.nn.functional.scaled_dot_product_attention=attention
        return self
    def __exit__(self,*args):
        torch.nn.functional.scaled_dot_product_attention=self.original
        for h in self.handles:h.remove()
    @property
    def flops(self):return self.linear+self.attention

@contextlib.contextmanager
def evaluation_weights(base):
    # FP32 parameters stay untouched. Allow AMP to cache frozen linear casts
    # inside no_grad, then restore every trainability flag before training.
    flags=[(p,p.requires_grad) for p in base.parameters()]
    for p,_ in flags:p.requires_grad_(True)
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):yield
    finally:
        for p,flag in flags:p.requires_grad_(flag)

def compat_tiny(base):
    module=sys.modules[base.__class__.__module__]
    for cls in (module.HuginnDynamicCache,module.HuginnStaticCache):
        for field in ('key_cache','value_cache'):
            setattr(cls,field,property(lambda self,f=field:getattr(self,'_native_'+f),lambda self,value,f=field:setattr(self,'_native_'+f,value)))

def generate(base,kind,prefixes,depth,latent,eos,limit,deadline,should_stop):
    n=len(prefixes);size=max(map(len,prefixes));x=torch.full((n,size),eos,device='cuda',dtype=torch.long);mask=torch.zeros_like(x)
    for i,p in enumerate(prefixes):x[i,-len(p):]=torch.tensor(p,device='cuda');mask[i,-len(p):]=1
    if raven(kind):
        assert len(set(map(len,prefixes)))==1
        cache=tiny_cache(base,depth,n,size+limit)
    else:
        from transformers.cache_utils import DynamicCache
        cache=DynamicCache();cache.append_new_layers(depth*base.config.num_hidden_layers-1)
        base.model.total_ut_steps=depth;base.config.total_ut_steps=depth;base.model.config.total_ut_steps=depth
    answers=[[] for _ in prefixes];ended=[False]*n;completion=[None]*n;ttft=None
    torch.cuda.synchronize();begun=time.perf_counter()
    for step in range(limit):
        if should_stop and should_stop():raise InterruptedError('Generation interrupted')
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Generation deadline')
        if raven(kind):
            start=0 if not step else size+step-1
            out=base(x,input_states=latent[:,start:start+x.shape[1]],num_steps=[0,depth],past_key_values=cache,use_cache=True,cache_position=torch.arange(start,start+x.shape[1],device='cuda'),output_details=DETAILS)
        else:
            pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0)
            if step:pos=pos[:,-1:]
            out=base(x,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,exit_at_step=depth-1,use_weighted_exit=False,logits_to_keep=1)
        logits=out.logits[:,-1]
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite logits')
        tokens=logits.argmax(-1);cache=out.past_key_values
        values=tokens.tolist();now=time.perf_counter()-begun
        if ttft is None:ttft=now
        for i,t in enumerate(values):
            if not ended[i]:
                answers[i].append(t)
                if t==eos:ended[i]=True;completion[i]=now
        del out,logits
        if all(ended):break
        x=tokens[:,None];mask=torch.cat([mask,torch.ones(n,1,device='cuda',dtype=mask.dtype)],dim=1)
    torch.cuda.synchronize();elapsed=time.perf_counter()-begun
    return answers,ended,{'group_latency_seconds':elapsed,'ttft_seconds':ttft,'completion_seconds':[v if v is not None else elapsed for v in completion]}

def eval_generation(model, rows, tokenizer, depths, amp_dtype, seed, eos, max_new_tokens,
                    output, step, family, deadline=None, batch_size=4, should_stop=None):
    assert max_new_tokens==512,'Matched boundary protocol retains512-token cap'
    before=model.training;flags=[p.requires_grad for p in model.base.parameters()];rng=torch.cuda.get_rng_state();result={}
    model.eval();base=model.base;kind=model.kind
    if raven(kind):compat_tiny(base)
    prefixes=[tokenizer.encode(row['prompt'],add_special_tokens=False) for row in rows]
    groups=groups_for(kind,prefixes,batch_size)
    params=sum(p.numel() for p in base.parameters())
    try:
        with evaluation_weights(base), (output/'predictions.jsonl').open('a',buffering=1) as f:
            for depth in depths:
                begun=time.monotonic();predictions=[];flops=0;linear_flops=0;attention_flops=0;ttfts=[]
                for group in groups:
                    ps=[prefixes[i] for i in group]
                    latent=torch.cat([states(base,kind,rows[i],len(prefixes[i])+max_new_tokens) for i in group]) if raven(kind) else None
                    with ComputeCounter(base) as counter:
                        answers,ended,timing=generate(base,kind,ps,depth,latent,eos,max_new_tokens,deadline,should_stop)
                    flops+=counter.flops;linear_flops+=counter.linear;attention_flops+=counter.attention;ttfts.append(timing['ttft_seconds'])
                    for j,i in enumerate(group):
                        row=rows[i];text=tokenizer.decode(answers[j],skip_special_tokens=True)
                        first=text.split('\n\nQuestion:',1)[0] if raven(kind) else text
                        score=grade(text,row['expected'],ended[j])
                        rec={'family':family,'step':step,'depth':depth,'index':row['index'],'question_sha256':row['question_sha256'],'question':row['question'],'expected':row['expected'],'text':text,'generated_tokens':len(answers[j]),'first_response_numerically_correct':grade(first,row['expected'],ended[j])['numerically_correct'],'new_question_continuation':first!=text,'latency_seconds':timing['completion_seconds'][j],'ttft_seconds':timing['ttft_seconds'],'allocated_group_flops':counter.flops/n if (n:=len(group)) else 0,'flops_counting':'linear matmuls + dense SDPA matmuls; excludes elementwise/norm/softmax','group_size':len(group),**score}
                        f.write(json.dumps(rec)+'\n');f.flush();predictions.append(rec)
                    del latent
                    print(json.dumps({'event':'generation_progress','family':family,'step':step,'depth':depth,'examples':len(predictions),'elapsed_seconds':time.monotonic()-begun}),flush=True)
                seconds=time.monotonic()-begun;n=len(predictions);tokens=sum(p['generated_tokens'] for p in predictions)
                result[str(depth)]={'examples':n,'correct':sum(p['correct'] for p in predictions),'accuracy':sum(p['correct'] for p in predictions)/n,'numerical_accuracy':sum(p['numerically_correct'] for p in predictions)/n,'first_response_numerical_accuracy':sum(p['first_response_numerically_correct'] for p in predictions)/n,'new_question_rate':sum(p['new_question_continuation'] for p in predictions)/n,'malformed_rate':sum(p['malformed'] for p in predictions)/n,'truncation_rate':sum(p['truncated'] for p in predictions)/n,'eos_rate':sum(p['eos'] for p in predictions)/n,'legacy_gsm_marker_rate':sum(p['parse_status']=='gsm_marker' for p in predictions)/n,'generated_tokens':tokens,'mean_generated_tokens':tokens/n,'generation_seconds':seconds,'seconds_per_example':seconds/n,'mean_latency_seconds':sum(p['latency_seconds'] for p in predictions)/n,'mean_ttft_seconds':sum(ttfts)/len(ttfts),'batch_size':batch_size,'counted_flops':flops,'linear_matmul_flops':linear_flops,'dense_attention_matmul_flops':attention_flops,'flops_per_example':flops/n,'flops_per_generated_token':flops/max(1,tokens),'flops_per_parameter_per_generated_token':flops/params/max(1,tokens),'physical_parameters':params}
    finally:
        model.train(before);torch.cuda.set_rng_state(rng)
        assert [p.requires_grad for p in base.parameters()]==flags,'Evaluation changed freeze scope'
    return result
