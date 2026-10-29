"""Inference-only paired base-model evaluation; independent fixed Tiny latent states.
No optimizer, backward, or checkpoint updates. Each record is flushed before logging.
"""
import argparse, gc, hashlib, json, os, time, traceback
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from answer_extraction import grade
from qwen_protocol import SYSTEM
MODELS={'tinyllama':('smcleish/Recurrent-TinyLlama-3T-train-recurrence-4','80ae03166c4a3783ae74c8c17226974790a1dd19'), 'ouro':('ByteDance/Ouro-1.4B-Thinking','3aaa2224253a92ca45cf2e3d427c360e1ef9c93d')}
DEPTHS=(1,2,4,8)
DETAILS={'return_logits':True,'return_latents':False,'return_head':False,'return_stats':False}

def raven(kind):
 return kind in ('tinyllama','olmo')


def states(model,kind,row,length):
 if not raven(kind):return None
 seed=(2468+int(row['question_sha256'][:12],16))%(2**63-1)
 with torch.random.fork_rng(devices=[0]):
  torch.cuda.manual_seed(seed)
  # Preserve native FP32 initialization distribution; seed and positions shared across depths.
  return model.initialize_state(torch.empty(1,length,model.config.n_embd,device='cuda',dtype=torch.float32))


def attention_heads(config):
 """Use num_attention_heads when it is a positive count.

 OLMo JSON omits that key and RavenConfig maps it to n_heads. A positive
 explicit count wins, so Tiny and recurrence-16 keep the heads they set.
 """
 heads=getattr(config,'num_attention_heads',None)
 if type(heads) is int and heads>0:return heads
 heads=getattr(config,'n_heads',None)
 if type(heads) is int and heads>0:return heads
 raise ValueError('Raven config has no attention head count')

def tiny_cache(model,depth,batch,length):
 # Preserve pinned static-cache update/lookup; initialize its tensor storage
 # without its obsolete superclass constructor (Transformers4.55 Cache API).
 import sys
 cls=sys.modules[model.__class__.__module__].HuginnStaticCache
 cache=cls.__new__(cls)
 cache._seen_tokens=0;cache.max_length=length;cache.lookup_strategy='full'
 cache.max_num_steps=model.config.n_layers_in_prelude+depth*model.config.n_layers_in_recurrent_block+model.config.n_layers_in_coda
 query_heads=attention_heads(model.config)
 shape=(cache.max_num_steps,batch,model.config.num_key_value_heads,length,model.config.n_embd//query_heads)
 cache.key_cache=torch.zeros(shape,device='cuda',dtype=torch.float32)
 cache.value_cache=torch.zeros(shape,device='cuda',dtype=torch.float16)
 cache.valid_mask=torch.zeros(cache.max_num_steps,length,device='cuda',dtype=torch.bool)
 return cache



def groups_for(kind,prefixes,batch):
 if kind=='ouro':return [list(range(i,min(i+batch,len(prefixes)))) for i in range(0,len(prefixes),batch)]
 from collections import defaultdict
 buckets=defaultdict(list)
 for i,p in enumerate(prefixes):buckets[len(p)].append(i)
 return [v[i:i+batch] for v in buckets.values() for i in range(0,len(v),batch)]

