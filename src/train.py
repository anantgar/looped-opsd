"""Native Tiny recurrent SFT; original optimizer, schedule and evaluation loop."""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import recovery


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)






def forward_kl(student_logits, teacher_logits):
    teacher_logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    return (teacher_logp.exp() * (teacher_logp-student_logp)).sum(-1)


def objective(logits, targets, maximum, student, mode, lam):
    ce = F.cross_entropy(logits[maximum], targets, reduction='sum')
    stats = {'ce_deep_sum': ce.detach().item()}
    if mode == 'sft':
        return ce, stats
    ce_s = F.cross_entropy(logits[student], targets, reduction='sum')
    if mode == 'dual_ce':
        stats.update(ce_student_sum=ce_s.detach().item())
        return .5*(ce+ce_s), stats
    if mode in ('sampled_ce', 'sampled_ce_constant'):
        stats.update(ce_student_sum=ce_s.detach().item(), weighted_ce_student_sum=lam*ce_s.detach().item())
        return ce+lam*ce_s, stats
    if mode == 'deep_ce':
        stats.update(ce_student_sum=ce_s.detach().item())
        return ce+ce_s, stats
    kd = forward_kl(logits[student], logits[maximum]).sum()
    stats.update(ce_student_sum=ce_s.detach().item(), kl_sum=kd.detach().item())
    if mode == 'kd':
        return ce+kd, stats
    if mode == 'ilsd':
        return ce+lam*ce_s+(1-lam)*kd, stats
    raise ValueError(mode)




def batch_tensors(examples, device, pad):
    length = max(len(x) for x,y in examples)
    # Right padding cannot influence preceding valid tokens under causal attention.
    x = torch.full((len(examples),length), pad, device=device, dtype=torch.long)
    y = torch.full_like(x,-100)
    for i,(inp,lab) in enumerate(examples):
        x[i,:len(inp)] = torch.tensor(inp,device=device)
        y[i,:len(lab)] = torch.tensor(lab,device=device)
    return x,y


def eval_nll(model, examples, depths, amp_dtype, seed, pad, batch=1):
    if batch < 1: raise ValueError('Invalid NLL batch')
    was_training = model.training
    model.eval()
    result = {}
    # Keep evaluation RNG from changing training state/data/depth sampling.
    # Right-padded causal batches match one-example forwards at labeled positions.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
        for depth in depths:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            total, n, text_total, text_n, eos_total, eos_n = 0.,0,0.,0,0.,0
            for start in range(0, len(examples), batch):
                x,y = batch_tensors(examples[start:start+batch],'cuda',pad)
                positions = y != -100
                with torch.autocast('cuda',dtype=amp_dtype):
                    z = model(x,[depth],positions)[depth]
                labels = y[positions]
                losses = F.cross_entropy(z,labels,reduction='none')
                text_mask = labels!=pad  # pad is the verified EOS ID; actual pads have label -100.
                total += losses.sum().item()
                n += positions.sum().item()
                text_total += losses[text_mask].sum().item()
                text_n += text_mask.sum().item()
                eos_total += losses[~text_mask].sum().item()
                eos_n += (~text_mask).sum().item()
            result[str(depth)] = {'answer_nll':total/n,'answer_tokens':n,'examples':len(examples),
                'answer_text_nll':text_total/max(text_n,1),'eos_nll':eos_total/max(eos_n,1)}
    model.train(was_training)
    return result






def run(cfg, args):
    cfg.setdefault('optimizer', 'adamw8')
    if cfg['mode'] == 'adaptive_ce':
        assert cfg['gradient_target_ratio'] == .25 and cfg['gradient_alpha_cap'] == 1.
        assert cfg['student_ce_schedule'] in ('constant-1','linear-1-to-0')
        assert cfg['maximum_depth'] == 4 and cfg['model_family'] == 'native-recurrent'
    if cfg['optimizer'] not in ('adamw8', 'adamw'):
        raise ValueError('Unknown optimizer; specify adamw8 or adamw explicitly')
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if getattr(args, 'resume', None) is None and (out/'metrics.jsonl').exists():
        raise ValueError('Existing run: use --resume instead of appending fresh training')
    if getattr(args, 'resume', None) and args.profile_only:
        raise ValueError('Profiles cannot resume')
    if getattr(args, 'resume', None) and (out/'config.json').exists():
        if json.loads((out/'config.json').read_text()) != cfg:
            raise ValueError('Resume config differs from the existing run')
    else:
        (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    assert cfg['kd_temperature']==1.0, 'Only temperature-1 KL is implemented'
    assert not cfg['feature_matching'], 'Hidden-state matching is outside this study'
    seed_all(cfg['seed'])
    torch.set_num_threads(4)
    device_name = torch.cuda.get_device_name()
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
    if cfg.get('amp_dtype') == 'float16': amp_dtype = torch.float16
    assert cfg['mode']=='sft' and cfg['native_kind']=='tinyllama'
    assert cfg['model_family']=='native-recurrent' and cfg['task']=='gsm8k'
    is_qwen = is_gsm = True
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from native_recurrent import NativeReadout, prepare_data, smoke_checks
    from native_recipe_eval import eval_generation
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True, trust_remote_code=True)
    enc, rows, pad = prepare_data(args.data_dir, tokenizer, cfg, out)
    base = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='sdpa', trust_remote_code=True).cuda()
    model = NativeReadout(base, cfg['native_kind'], cfg['trainable_scope'], cfg['native_training_readout'])
    model.evaluation_protocol = cfg['evaluation_protocol']
    model.extraction_protocol = cfg['protocol']
    checks = smoke_checks
    frozen_initial = model.frozen_digest() if is_qwen and cfg.get('trainable_scope','full') != 'full' else None
    trainable = [p for p in model.parameters() if p.requires_grad]
    params = sum(p.numel() for p in model.parameters())
    smoke_examples = sorted(enc['train'],key=lambda example:len(example[0]),reverse=True) if is_gsm else enc['train']
    checks(model,smoke_examples,amp_dtype,pad,out)
    seed_all(cfg['seed'])
    metadata = {'gpu':device_name,'torch':torch.__version__,'parameters':params,
                'trainable_parameters':sum(p.numel() for p in trainable),'trainable_scope':cfg.get('trainable_scope','full'),
                'amp_dtype':str(amp_dtype),'experiment_commit':os.environ.get('EXPERIMENT_COMMIT'),'full_gradient_loops':cfg['maximum_depth'],
                'upstream_commit':cfg['upstream_commit']}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'event':'loaded','metadata':metadata}),flush=True)
    if cfg.get('optimizer') == 'adamw8':
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=(.9,.95),eps=1e-8)
    else:
        opt = torch.optim.AdamW(trainable,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],
                           betas=(0.9,0.95),eps=1e-8,foreach=False if is_qwen else None)
    scaler = torch.amp.GradScaler('cuda',enabled=amp_dtype == torch.float16,init_scale=128)
    total_steps = cfg['epochs'] * math.ceil(len(enc['train'])/cfg['effective_batch'])
    step = 0
    token_count = 0
    target_tokens = 0
    skipped_steps = 0
    elapsed_train = 0.
    start = time.monotonic()
    deadline = start+cfg['max_runtime_seconds']
    stopped = [False]
    signal.signal(signal.SIGTERM,lambda *a: stopped.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *a: stopped.__setitem__(0,True))
    depth_rng = random.Random(cfg['seed']+100)
    cursor = recovery.DataCursor(len(enc['train']), cfg['effective_batch'], cfg['epochs'], cfg['seed']+200)
    identity = recovery.fingerprint(cfg, out/'split_manifest.json', metadata)
    progress = {'step':0, 'input_tokens':0, 'target_tokens':0, 'skipped_updates':0,
        'train_seconds':0., 'wall_seconds':0., 'pending':['initial_nll','initial_generation'],
        'status':'training', 'validation':None, 'initial_mc':None, 'final_val':None,
        'final_mc':None, 'test_metrics':None, 'tracking':None, 'metrics_bytes':0, 'predictions_bytes':0}
    if getattr(args, 'resume', None):
        cursor_state, progress = recovery.load(args.resume, model.base, opt, scaler, depth_rng, identity)
        cursor = recovery.DataCursor(len(enc['train']),cfg['effective_batch'],cfg['epochs'],cfg['seed']+200,cursor_state)
        step, token_count, target_tokens = progress['step'], progress['input_tokens'], progress['target_tokens']
        skipped_steps, elapsed_train = progress['skipped_updates'], progress['train_seconds']
        # Roll back partially emitted evaluation events to the durable checkpoint.
        for name, key in [('metrics.jsonl','metrics_bytes'),('predictions.jsonl','predictions_bytes')]:
            path = out/name
            if path.exists():
                if path.stat().st_size < progress[key]:
                    raise ValueError('Recovered log is shorter than durable checkpoint')
                with path.open('r+b') as stream: stream.truncate(progress[key])
            elif progress[key]:
                raise ValueError('Restore matching logs alongside the recovery checkpoint')
        for path in out.glob('checkpoint-step*.pt'):
            if int(path.stem.split('step')[1]) > step: path.unlink()
        crossing = out/'crossing.json'
        if crossing.exists() and json.loads(crossing.read_text())['step'] > step:
            crossing.unlink(); (out/'crossing.pt').unlink(missing_ok=True)
        (out/'failure.json').unlink(missing_ok=True)
    prior_wall = progress['wall_seconds']
    deadline = start + max(0.,cfg['max_runtime_seconds']-prior_wall)
    tracker = None
    if not args.profile_only and os.environ.get('WANDB_MODE', 'offline') != 'disabled':
        import wandb
        tracking = progress.get('tracking')
        tracker = wandb.init(project=os.environ.get('WANDB_PROJECT', 'depth-opd'), entity=os.environ.get('WANDB_ENTITY'),
                             id=tracking['run_id'] if tracking else None,
                             resume='must' if tracking and os.environ.get('WANDB_MODE')=='online' else None,
                             name=out.name, config={**cfg, **metadata}, dir=str(out),
                             mode=os.environ.get('WANDB_MODE', 'offline'))
        tracker.define_metric('step')
        tracker.define_metric('*',step_metric='step')
        progress['tracking'] = {'run_id':tracker.id,'url':tracker.url,'mode':os.environ.get('WANDB_MODE','offline')}
        (out/'tracking.json').write_text(json.dumps(progress['tracking'],indent=2)+'\n')
    log = (out/'metrics.jsonl').open('a',buffering=1)
    def generation(data_rows, depths, family, at_step):
        prediction_path = out/'predictions.jsonl'
        prediction_offset = prediction_path.stat().st_size if prediction_path.exists() else 0
        metrics = eval_generation(model,data_rows,tokenizer,depths,amp_dtype,cfg['eval_seed'],
            pad,cfg['max_new_tokens'],out,at_step,family,deadline,batch_size=cfg.get('generation_batch_size',16),should_stop=lambda:stopped[0])
        emit({'event':family,'step':at_step,'metrics':metrics})
        if tracker is not None and cfg.get('trace_examples', 2):
            import wandb
            from trace_logging import trace_rows, COLUMNS
            traces = trace_rows(prediction_path, prediction_offset, data_rows, at_step,
                                family, cfg.get('trace_examples', 2))
            tracker.log({'step':at_step, 'traces/'+family:wandb.Table(columns=COLUMNS,data=traces)})
        return metrics
    diagnostic_events=[]
    def emit(event):
        if cfg.get('midway_overfit_check') and event['event'] in ('validation_generation','train_generation','interim_validation'):
            diagnostic_events.append(event)
        event['wall_seconds'] = prior_wall + time.monotonic()-start
        print(json.dumps(event),flush=True)
        log.write(json.dumps(event)+'\n')
        if tracker is not None:
            flat = {k:v for k,v in event.items() if isinstance(v,(int,float,bool))}
            for key,value in event.get('components',{}).items(): flat['train/'+key]=value
            family = event['event'] if is_gsm else ('interim_validation' if event['event']=='interim_validation' else 'validation')
            for depth,values in event.get('metrics',{}).items():
                for key,value in values.items(): flat[f'{family}/depth_{depth}/{key}']=value
            for family,depths in event.get('metrics_test',{}).items():
                for depth,values in depths.items():
                    for key,value in values.items(): flat[f'test/{family}/depth_{depth}/{key}']=value
            tracker.log(flat)
    validation_examples = enc['validation'][:8] if args.profile_only else enc['validation']
    if args.profile_only: total_steps = cfg['profile_steps']
    validation, initial_mc = progress['validation'], progress['initial_mc']
    final_val, final_mc, test_metrics = progress['final_val'], progress['final_mc'], progress['test_metrics']
    def check_midway():
        if not cfg.get('midway_overfit_check') or args.profile_only or step!=total_steps//2 or progress['pending'] or stopped[0]:return
        if (out/'midway-overfit-check.json').exists():return
        from native_study import midway_check
        audit=midway_check([json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()],str(cfg['maximum_depth']),step)
        (out/'midway-overfit-check.json').write_text(json.dumps(audit,indent=2)+'\n')
        if audit['overfit']:
            stopped[0]=True
            (out/'overfit-cutoff.json').write_text(json.dumps(audit,indent=2)+'\n')
            durable()
    def durable():
        if args.profile_only: return
        log.flush()
        os.fsync(log.fileno())
        predictions = out/'predictions.jsonl'
        if predictions.exists():
            with predictions.open('rb') as stream: os.fsync(stream.fileno())
        progress.update(step=step,input_tokens=token_count,target_tokens=target_tokens,
            skipped_updates=skipped_steps,train_seconds=elapsed_train,
            wall_seconds=prior_wall+time.monotonic()-start,
            validation=validation,initial_mc=initial_mc,final_val=final_val,
            final_mc=final_mc,test_metrics=test_metrics,
            metrics_bytes=(out/'metrics.jsonl').stat().st_size,
            predictions_bytes=predictions.stat().st_size if predictions.exists() else 0)
        recovery.save(out/'recovery.pt',model.base,opt,scaler,cursor,depth_rng,progress,identity)
    def evaluate_pending():
        nonlocal validation, initial_mc, final_val, final_mc, test_metrics
        while progress['pending'] and not stopped[0] and time.monotonic()<deadline:
            task = progress['pending'][0]
            insert = []
            if task == 'initial_nll':
                validation=eval_nll(model,validation_examples,cfg['eval_depths'],amp_dtype,cfg['eval_seed'],pad,cfg.get('nll_batch',1))
                emit({'event':'validation','step':0,'metrics':validation})
            elif task == 'initial_generation':
                if is_gsm:
                    initial_mc=generation(rows['validation'][:cfg.get('generation_batch_size',16)] if args.profile_only else rows['validation'][:cfg.get('generation_validation_examples',len(rows['validation']))],
                        [cfg['maximum_depth']] if args.profile_only else cfg['eval_depths'],'validation_generation',0)
                    if not args.profile_only and cfg.get('generation_train_examples',0):
                        generation(rows['train'][:cfg['generation_train_examples']],cfg.get('train_eval_depths',[cfg['maximum_depth']]),'train_generation',0)
                elif not args.profile_only:
                    raise ValueError('Only GSM generation evaluation is supported')
                    emit({'event':'validation_multiple_choice','step':0,'metrics':initial_mc})
            elif task == 'interim':
                v=eval_nll(model,enc['validation'][:cfg['interim_validation_examples']],
                    cfg['eval_depths'],amp_dtype,cfg['eval_seed'],pad,cfg.get('nll_batch',1))
                emit({'event':'interim_validation','step':step,'metrics':v})
                save_checkpoint(out,model,opt,scaler,step,token_count)
                if is_gsm and cfg['mode'] in ('sft','sampled_ce','adaptive_ce','sampled_ce_constant') and cfg.get('save_validation_snapshots',True):
                    snapshot=out/f'checkpoint-step{step}.pt'
                    snapshot.unlink(missing_ok=True)
                    os.link(out/'checkpoint.pt',snapshot)
                crossing=out/'crossing.json'
                if is_gsm and cfg.get('quality_target') is not None and (not crossing.exists() or json.loads(crossing.read_text())['step']==step) and v[str(cfg['maximum_depth'])]['answer_text_nll']<=cfg['quality_target']:
                    (out/'crossing.pt').unlink(missing_ok=True)
                    os.link(out/'checkpoint.pt',out/'crossing.pt')
                    crossing.write_text(json.dumps({'step':step,'train_seconds':elapsed_train,'nll':v[str(cfg['maximum_depth'])]['answer_text_nll']})+'\n')
                    insert=['crossing_generation']
            elif task == 'crossing_generation': generation(rows['validation'],cfg['eval_depths'],task,step)
            elif task == 'periodic_generation':
                generation(rows['validation'][:cfg.get('generation_validation_examples',len(rows['validation']))],
                           cfg['eval_depths'],'validation_generation',step)
                if cfg.get('generation_train_examples',0):
                    generation(rows['train'][:cfg['generation_train_examples']],
                               cfg.get('train_eval_depths',[cfg['maximum_depth']]),'train_generation',step)
            elif task == 'final_nll':
                save_checkpoint(out,model,opt,scaler,step,token_count)
                final_val=eval_nll(model,validation_examples,cfg['eval_depths'],amp_dtype,cfg['eval_seed'],pad,cfg.get('nll_batch',1))
                emit({'event':'validation','step':step,'metrics':final_val})
            elif task == 'final_generation':
                if is_gsm:
                    final_mc=generation(rows['validation'][:cfg.get('final_generation_validation_examples',cfg.get('generation_validation_examples',len(rows['validation'])))],cfg['eval_depths'],'final_validation_generation',step)
                    if not args.profile_only and cfg.get('generation_train_examples',0):
                        generation(rows['train'][:cfg['generation_train_examples']],cfg.get('train_eval_depths',[cfg['maximum_depth']]),'train_generation',step)
                elif not args.profile_only:
                    raise ValueError('Only GSM generation evaluation is supported')
                    emit({'event':'validation_multiple_choice','step':step,'metrics':final_mc})
            elif task == 'test':
                test_metrics={'nll':eval_nll(model,enc['test'],cfg['eval_depths'],amp_dtype,cfg['eval_seed'],pad,cfg.get('nll_batch',1)),
                    'multiple_choice':None}
                emit({'event':'test','step':step,'metrics_test':test_metrics})
            else: raise ValueError(f'Unknown pending evaluation {task}')
            progress['pending']=insert+progress['pending'][1:]
            durable()
    try:
        durable()
        evaluate_pending()
        check_midway()
        while progress['status']=='training':
            indices=cursor.batch()
            if indices is None: break
            if stopped[0] or time.monotonic()>deadline or token_count>=cfg['max_input_tokens']:
                stopped[0]=True
                break
            examples=[enc['train'][i] for i in indices]
            denominator = sum(sum(v!=-100 for v in y) for x,y in examples)
            l = depth_rng.choice(cfg['student_depths'])
            exits = [cfg['maximum_depth']] if cfg['mode']=='sft' else [l,cfg['maximum_depth']]
            schedule_steps = cfg.get('profile_schedule_steps',total_steps) if args.profile_only else total_steps
            lam = (1. if cfg.get('student_ce_schedule') == 'constant-1'
                   else max(0.,1-step/max(schedule_steps-1,1)))
            warm = max(1,round(schedule_steps*cfg['warmup_fraction']))
            lr_factor = (step+1)/warm if step<warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/max(1,schedule_steps-warm)))
            for g in opt.param_groups: g['lr']=cfg['learning_rate']*lr_factor
            torch.cuda.synchronize()
            begun=time.monotonic()
            opt.zero_grad(set_to_none=True)
            totals={}
            balance = None
            balance_stats = {}
            if cfg['mode'] == 'adaptive_ce':
                from gradient_balance import GradientBalance
                balance = GradientBalance(trainable)
            for i in range(0,len(examples),cfg['microbatch']):
                micro=examples[i:i+cfg['microbatch']]
                x,y = batch_tensors(micro,'cuda',pad)
                positions = y!=-100
                with torch.autocast('cuda',dtype=amp_dtype,cache_enabled=cfg.get('amp_weight_cache',True)):
                    z=model(x,exits,positions)
                    if balance is None:
                        loss,parts=objective(z,y[positions],cfg['maximum_depth'],l,cfg['mode'],lam)
                    else:
                        deep_ce=F.cross_entropy(z[cfg['maximum_depth']],y[positions],reduction='sum')
                        student_ce=F.cross_entropy(z[l],y[positions],reduction='sum')
                        loss=deep_ce+student_ce
                        parts={'ce_deep_sum':deep_ce.detach().item(), 'ce_student_sum':student_ce.detach().item()}
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                if balance is None:
                    scaler.scale(loss/denominator).backward()
                else:
                    balance.backward(deep_ce/denominator,student_ce/denominator,scaler)
                for k,v in parts.items(): totals[k]=totals.get(k,0.)+v
            if balance is not None:
                balance_stats=balance.combine(scaler.get_scale(),lam,cfg['gradient_target_ratio'],
                    cfg['gradient_alpha_cap'],cfg['gradient_norm_epsilon'])
                totals['weighted_ce_student_sum']=balance_stats['alpha']*totals['ce_student_sum']
                del balance
            if cfg.get('native_kind')=='olmo' and torch.cuda.max_memory_allocated()/2**30 > cfg['profile_memory_limit_gib']:
                raise RuntimeError('Depth-16 batch 16 exceeded the memory gate; not shrinking the batch')
            scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad_norm'])
            if not torch.isfinite(norm):
                emit({'event':'nonfinite_gradient','step':step,'scale':scaler.get_scale()})
            old_scale=scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped= scaler.get_scale()<old_scale
            skipped_steps += int(skipped)
            if skipped_steps>cfg['max_skipped_steps']: raise RuntimeError('Too many skipped FP16 updates')
            torch.cuda.synchronize()
            seconds=time.monotonic()-begun
            elapsed_train+=seconds
            step+=1
            token_count+=sum(len(x) for x,y in examples)
            target_tokens+=denominator
            emit({'event':'train','step':step,'epoch':cursor.epoch+1,'mode':cfg['mode'],
                  'student_depth':l if cfg['mode']!='sft' else None,'lambda':lam,
                  'components':{k:v/denominator for k,v in totals.items()},
                  'step_seconds':seconds,'train_seconds':elapsed_train,
                  'input_tokens':token_count,'target_tokens':target_tokens,
                  'lr':opt.param_groups[0]['lr'],'grad_norm':float(norm),
                  'clip_multiplier':min(1.,cfg['clip_grad_norm']/max(float(norm),1e-30)),
                  **balance_stats,
                  'skipped':skipped,'optimizer_state_uint8':any(isinstance(v,torch.Tensor) and v.dtype==torch.uint8 for state in opt.state.values() for v in state.values()),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
            # Completed gradients are not part of recovery state or later evaluation.
            # Release them before NLL/generation to leave room for native KV caches.
            opt.zero_grad(set_to_none=True)
            cursor.advance(len(indices))
            progress['pending']=[]
            if not args.profile_only and step%cfg['validate_every']==0: progress['pending'].append('interim')
            if is_gsm and not args.profile_only and (step in cfg['generation_steps'] if 'generation_steps' in cfg else step%cfg['generation_every']==0) and (not is_qwen or step<total_steps): progress['pending'].append('periodic_generation')
            if progress['pending'] or step%cfg.get('recovery_every',8)==0 or stopped[0]: durable()
            evaluate_pending()
            check_midway()
            if args.profile_only and step>=cfg['profile_steps']:
                stopped[0]=True
                break
        if not stopped[0] and time.monotonic()>=deadline: stopped[0]=True
        if not stopped[0] or args.profile_only:
            if progress['status']=='training':
                progress['status']='finalizing'
                progress['pending']=['final_nll','final_generation']+(['test'] if not is_gsm and not args.profile_only else [])
                durable()
            # Profiles also perform final NLL despite their step cap.
            if args.profile_only: stopped[0]=False
            evaluate_pending()
            if progress['pending']: stopped[0]=True
            else: progress['status']='complete'
        durable()
        summary={'status':'profile_complete' if args.profile_only else ('overfit_stopped' if (out/'overfit-cutoff.json').exists() else ('budget_stopped' if stopped[0] else 'complete')),
                 'steps':step,'input_tokens':token_count,'target_tokens':target_tokens,
                 'train_seconds':elapsed_train,'wall_seconds':prior_wall+time.monotonic()-start,
                 'skipped_updates':skipped_steps,'initial_validation':validation,'final_validation':final_val,
                 'initial_validation_mc':None if args.profile_only or is_gsm else initial_mc,
                 'final_validation_mc':None if is_gsm else final_mc,
                 'initial_validation_generation':initial_mc if is_gsm else None,
                 'final_validation_generation':final_mc if is_gsm else None,'test':test_metrics}
        if frozen_initial is not None:
            frozen_final = model.frozen_digest()
            audit = {'initial_sha256':frozen_initial,'final_sha256':frozen_final,'unchanged':frozen_initial == frozen_final}
            (out/'frozen-weight-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
            assert audit['unchanged'], 'Frozen weights changed'
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    except BaseException as e:
        # Do not overwrite the last committed recovery state with a partial update.
        (out/'failure.json').write_text(json.dumps({'error':str(e),'steps':step,'recoverable_checkpoint':str(out/'recovery.pt')},indent=2)+'\n')
        raise
    finally:
        log.close()
        if tracker is not None: tracker.finish()


def save_checkpoint(out,model,opt,scaler,step,tokens):
    # Checkpoint the one shared model, not separate teacher/student copies.
    # Evaluation snapshots stay small; recovery.pt separately contains full training state.
    recovery.atomic_save({'model':model.base.state_dict(),'step':step,'input_tokens':tokens},out/'checkpoint.pt')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--tokenizer-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--profile-only',action='store_true')
    parser.add_argument('--resume',help='Restore a versioned recovery.pt with matching logs/config/runtime')
    args=parser.parse_args()
    with recovery.exclusive_run(args.output):
        run(json.loads(Path(args.config).read_text()),args)
