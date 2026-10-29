"""Vanilla native SFT on pinned, already recurrent HF checkpoints."""
import functools
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint


def raven_kind(kind):
    return kind in ('tinyllama', 'olmo')


class NativeReadout(torch.nn.Module):
    def __init__(self, base, kind, scope='full', training_readout='published-exit-mixture'):
        super().__init__(); self.base=base; self.kind=kind
        if training_readout not in ('published-exit-mixture','final-depth-ce'):raise ValueError(training_readout)
        self.training_readout=training_readout
        self.extraction_protocol='qwen-gsm-last-box-v2'
        for p in base.parameters(): p.requires_grad_(True)
        if scope not in ('full','lexical-frozen','lora','core'): raise ValueError(scope)
        if scope=='lexical-frozen':
            base.get_input_embeddings().requires_grad_(False)
            base.lm_head.requires_grad_(False)
        if scope=='core':
            for p in base.parameters():p.requires_grad_(False)
            if raven_kind(kind):
                base.transformer.core_block.requires_grad_(True);base.transformer.adapter.requires_grad_(True)
            else:
                base.model.layers.requires_grad_(True);base.model.norm.requires_grad_(True)
        if scope=='lora':
            import math
            for p in base.parameters():p.requires_grad_(False)
            class Adapter(torch.nn.Module):
                def __init__(self,linear):
                    super().__init__();self.linear=linear
                    self.a=torch.nn.Parameter(torch.empty(16,linear.in_features,device=linear.weight.device))
                    self.b=torch.nn.Parameter(torch.zeros(linear.out_features,16,device=linear.weight.device))
                    torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
                def forward(self,x):
                    return self.linear(x)+torch.nn.functional.linear(torch.nn.functional.linear(x,self.a),self.b)
            for name,module in list(base.named_modules()):
                if isinstance(module,torch.nn.Linear) and name.split('.')[-1] in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','Wqkv','proj','fc','gate','w1','w2','w3'):
                    parent,_,leaf=name.rpartition('.')
                    setattr(base.get_submodule(parent),leaf,Adapter(module))
            assert any(p.requires_grad for p in base.parameters()), 'No supported LoRA targets'
        # Freeze the unused Ouro gate when CE uses only the final forced exit.
        if kind=='ouro':base.model.early_exit_gate.requires_grad_(training_readout!='final-depth-ce' and scope in ('full','core','lexical-frozen'))
        self.checkpoint_layers=False
        blocks=(list(base.transformer.prelude)+list(base.transformer.core_block)+list(base.transformer.coda)
                if raven_kind(kind) else list(base.model.layers))
        for block in blocks:
            original=block.forward
            @functools.wraps(original)
            def wrapped(*args, _original=original, **kwargs):
                if self.training and self.checkpoint_layers and torch.is_grad_enabled():
                    # Raven mutates the CPU block-index counter between layers.
                    saved=tuple(a.clone() if isinstance(a,torch.Tensor) and a.device.type=='cpu' else a for a in args)
                    return checkpoint(_original,*saved,use_reentrant=False,preserve_rng_state=True,**kwargs)
                return _original(*args,**kwargs)
            block.forward=wrapped

    def native_logits(self,tokens,depth,keep=0):
        if raven_kind(self.kind):
            # A scalar num_steps runs the whole core under no_grad upstream.
            # [no-grad steps, grad steps] explicitly requests full BPTT.
            return self.base(tokens,num_steps=[0,depth],use_cache=False,
                output_details={'return_logits':True,'return_latents':False,'return_head':False,'return_stats':False}).logits.float()
        self.base.model.total_ut_steps=depth
        self.base.model.config.total_ut_steps=depth
        self.base.config.total_ut_steps=depth
        if self.training and self.training_readout=='published-exit-mixture':
            # Labels normally select expected exit logits in the published code.
            # Request those same logits, but compute our explicitly shifted/masked CE.
            saved=self.base.early_exit_threshold;self.base.early_exit_threshold=None
            try:
                return self.base(tokens,use_cache=False,exit_at_step=-1,
                                 use_weighted_exit=True,exit_threshold=None,logits_to_keep=keep).logits.float()
            finally:self.base.early_exit_threshold=saved
        return self.base(tokens,use_cache=False,exit_at_step=depth-1,
                         use_weighted_exit=False,logits_to_keep=keep).logits.float()

    def forward(self,tokens,exits,positions=None):
        depths=sorted(set(exits))
        if len(depths)==1:
            z=self.native_logits(tokens,depths[0])
            return {depths[0]:z if positions is None else z[positions]}
        if len(depths)>2:
            if not raven_kind(self.kind) or self.training_readout!='final-depth-ce':
                raise ValueError('Multiple Raven exits require final-depth-ce')
            return self._raven_exits(tokens,depths,positions)
        assert len(depths)==2 and self.training_readout=='final-depth-ce'
        student,maximum=depths
        assert 1<=student<maximum
        if self.kind=='ouro':
            self.base.model.total_ut_steps=maximum
            self.base.model.config.total_ut_steps=maximum
            self.base.config.total_ut_steps=maximum
            _,hidden,_=self.base.model(tokens,use_cache=False)
            result={d:self.base.lm_head(hidden[d-1]).float() for d in depths}
            return result if positions is None else {d:z[positions] for d,z in result.items()}
        if not raven_kind(self.kind):raise ValueError(self.kind)
        # Capture the student's latent in the SAME recurrence trajectory.
        # Upstream predict_from_latents adds an extra norm before the coda;
        # reproduce the ordinary forward's coda directly instead.
        captured={};original=self.base.core_block_forward
        def capture(*args,**kwargs):
            result=original(*args,**kwargs)
            current=kwargs.get('current_step',args[6] if len(args)>6 else None)
            if int(current)+1==student:
                captured.update(latent=result[0],freqs=args[2],mask=args[3])
            return result
        self.base.core_block_forward=capture
        try:z=self.native_logits(tokens,maximum)
        finally:self.base.core_block_forward=original
        x=captured['latent'];block_idx=torch.tensor(0,device='cpu',dtype=torch.long)
        for block in self.base.transformer.coda:
            block_idx-=1
            x=block(x,captured['freqs'],block_idx,captured['mask'],None)
        x=self.base.transformer.ln_f(x)
        shallow=self.base.lm_head(x).float()
        if positions is not None:shallow=shallow[positions]
        return {maximum:z if positions is None else z[positions],student:shallow}

    def _raven_exits(self,tokens,depths,positions):
        """One recurrence to the deepest exit. Shallower exits reuse that trajectory.

        The coda is the ordinary forward's coda. predict_from_latents adds an
        extra norm and is not used. num_steps is [0, maximum], so every loop
        stays in the gradient region.
        """
        maximum=depths[-1]
        if any(depth<1 or depth>maximum for depth in depths):
            raise ValueError('Exit depth outside the recurrence')
        wanted=set(depths);captured={};original=self.base.core_block_forward
        def capture(*args,**kwargs):
            result=original(*args,**kwargs)
            current=kwargs.get('current_step',args[6] if len(args)>6 else None)
            step=int(current)+1
            if step in wanted and step!=maximum:
                captured[step]=(result[0],args[2],args[3])
            return result
        self.base.core_block_forward=capture
        try:final=self.native_logits(tokens,maximum)
        finally:self.base.core_block_forward=original
        if positions is not None:final=final[positions]
        out={maximum:final}
        for depth in depths:
            if depth==maximum:continue
            if depth not in captured:raise RuntimeError(f'Recurrence did not visit exit {depth}')
            latent,freqs,mask=captured[depth];x=latent
            block_idx=torch.tensor(0,device='cpu',dtype=torch.long)
            for block in self.base.transformer.coda:
                block_idx-=1
                x=block(x,freqs,block_idx,mask,None)
            x=self.base.transformer.ln_f(x)
            logits=self.base.lm_head(x).float()
            if positions is not None:logits=logits[positions]
            out[depth]=logits
        return out

    def cached_generate(self,prefixes,depth,eos,amp_dtype,max_new_tokens,deadline=None,should_stop=None):
        import time
        assert self.kind=='ouro' and not self.training
        self.base.model.total_ut_steps=depth
        self.base.model.config.total_ut_steps=depth
        self.base.config.total_ut_steps=depth
        size=max(map(len,prefixes));n=len(prefixes)
        x=torch.full((n,size),eos,device='cuda',dtype=torch.long)
        mask=torch.zeros_like(x)
        for i,prefix in enumerate(prefixes):
            x[i,-len(prefix):]=torch.tensor(prefix,device='cuda');mask[i,-len(prefix):]=1
        from transformers.cache_utils import DynamicCache
        # Transformers4.55's cache API needs96 distinct layer/pass slots.
        # The pinned upstream legacy cache constructor is incompatible with it.
        cache=DynamicCache();cache.append_new_layers(depth*self.base.config.num_hidden_layers-1)
        answers=[[] for _ in prefixes];ended=[False]*n
        for step in range(max_new_tokens):
            if should_stop is not None and should_stop():raise InterruptedError('Generation interrupted')
            if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Generation deadline')
            positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,0)
            if step:positions=positions[:,-1:]
            with torch.autocast('cuda',dtype=amp_dtype):
                out=self.base(x,attention_mask=mask,position_ids=positions,
                    past_key_values=cache,use_cache=True,exit_at_step=depth-1,
                    use_weighted_exit=False,logits_to_keep=1)
            cache=out.past_key_values;tokens=out.logits[:,-1].argmax(-1)
            for i,t in enumerate(tokens.tolist()):
                if not ended[i]:answers[i].append(t);ended[i]=t==eos
            if all(ended):break
            x=tokens[:,None];mask=torch.cat([mask,torch.ones((n,1),device='cuda',dtype=mask.dtype)],dim=1)
        return answers,ended

    def frozen_digest(self):
        h=hashlib.sha256()
        for n,p in self.base.named_parameters():
            if not p.requires_grad:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


def prepare_data(data_dir,tokenizer,cfg,output):
    from qwen_gsm import prepare_data as prepare_qwen
    from qwen_protocol import SYSTEM
    # Reuse the pinned identities and label/masking audit; change only rendering.
    class Renderer:
        eos_token_id=tokenizer.eos_token_id
        def convert_tokens_to_ids(self,t):
            return self.eos_token_id if t=='<|im_end|>' else tokenizer.convert_tokens_to_ids(t)
        def encode(self,*args,**kwargs):return tokenizer.encode(*args,**kwargs)
        def apply_chat_template(self,messages,**kwargs):
            if cfg['native_kind']=='ouro':return tokenizer.apply_chat_template(messages,enable_thinking=True,**kwargs)
            question=messages[-1]['content']
            # Raw-text checkpoint: fixed synthetic demonstration, no GSM leakage.
            return (tokenizer.bos_token+SYSTEM+'\n\nQuestion: Mia has 3 apples and gets 2 more. How many apples does Mia have?'
                    '\nAnswer: 3 + 2 = 5.\\boxed{5}\n\nQuestion: '+question+'\nAnswer:\n')
    return prepare_qwen(data_dir,Renderer(),cfg,output)


def smoke_checks(model,examples,amp_dtype,pad,output):
    from train import batch_tensors,seed_all
    # Short causal/identity/gradient audit; memory is profiled separately on real batches.
    x=torch.tensor(examples[0][0][:24],device='cuda').unsqueeze(0)
    d=4;model.eval();seed_all(2468)
    with torch.no_grad(),torch.autocast('cuda',dtype=amp_dtype):
        reference=model.native_logits(x,d)
        seed_all(2468);got=model(x,[d])[d]
        assert torch.equal(reference,got),'Native logits changed'
        changed=x.clone();changed[0,-1]=(changed[0,-1]+1)%model.base.config.vocab_size
        seed_all(2468);altered=model(changed,[d])[d]
        assert torch.equal(got[:,:-1],altered[:,:-1]),'Causal leakage'
    if model.kind=='ouro':
        with torch.no_grad():
            prefixes=[x[0].tolist(),x[0,:19].tolist()]
            cached,_=model.cached_generate(prefixes,d,2,amp_dtype,8)
            dense=[]
            for prefix in prefixes:
                seq=prefix[:];answer=[]
                for _ in range(8):
                    with torch.autocast('cuda',dtype=amp_dtype):
                        token=model.native_logits(torch.tensor([seq],device='cuda'),d)[0,-1].argmax().item()
                    answer.append(token);seq.append(token)
                    if token==2:break
                dense.append(answer)
            assert cached==dense,'Native KV cache changed greedy continuation'
        model.train()
        with torch.no_grad(),torch.autocast('cuda',dtype=amp_dtype):
            dense=model.native_logits(x,d);mark=torch.zeros_like(x,dtype=torch.bool);mark[:,-5:]=True
            sparse=model(x,[d],mark)[d]
            assert torch.equal(sparse,dense[mark]),'Native training readout changed'
            if model.training_readout=='final-depth-ce':
                direct=model.base(x,use_cache=False,exit_at_step=d-1,use_weighted_exit=False).logits.float()
                assert torch.equal(dense,direct),'Final CE is not the forced final native exit'
                assert not model.base.model.early_exit_gate.weight.requires_grad,'Unused exit gate must be frozen'
    model.train();model.checkpoint_layers=True;seed_all(2468)
    with torch.autocast('cuda',dtype=amp_dtype):z=model(x,[d])[d];loss=z[:,-1].log_softmax(-1)[0,3].neg()
    loss.backward()
    core=(model.base.transformer.core_block if raven_kind(model.kind) else model.base.model.layers)
    groups=[any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in layer.parameters()) for layer in core]
    assert all(groups),'Missing/nonfinite core layer gradients'
    if model.kind=='ouro' and model.base.model.early_exit_gate.weight.requires_grad:
        assert model.base.model.early_exit_gate.weight.grad is not None
        assert model.base.model.early_exit_gate.weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True)
    (output/'checks.json').write_text(json.dumps({'native_final_exit_exact':True,'causal_mask':True,
        'native_cached_greedy_checked':model.kind=='ouro','native_weighted_training_logits_exact':model.kind=='ouro' and model.training_readout=='published-exit-mixture','training_readout':model.training_readout,'all_core_layers_receive_gradient':True,'full_bptt_four_loops':True,'activation_checkpointing':True},indent=2)+'\n')
