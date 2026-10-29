"""Original pinned GSM questions/solutions, retokenized with Qwen native chat."""
import hashlib
import json
import re
import time
from pathlib import Path
import torch
from qwen_protocol import VERSION, SYSTEM, prompt, grade, final_answer


def prepare_data(data_dir, tokenizer, cfg, output):
    manifest = json.loads(Path(cfg['split_manifest']).read_text())
    profile = json.loads(Path(cfg['data_profile']).read_text())
    raw = {}
    for source in ['train', 'test']:
        path = Path(data_dir) / (source+'.jsonl')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == profile['statistics'][source]['sha256']
        raw[source] = [json.loads(line) for line in path.read_text().splitlines()]
    enc, rows, seen, lengths = {}, {}, set(), {}
    eos = tokenizer.eos_token_id
    assert eos == tokenizer.convert_tokens_to_ids('<|im_end|>')
    for split in ['train', 'validation', 'test']:
        enc[split], rows[split], lengths[split] = [], [], []
        assert len(manifest[split]) == cfg[split+'_examples']
        for item in manifest[split]:
            row = dict(raw['test' if split == 'test' else 'train'][item['index']])
            question = row['question'].strip()
            digest = hashlib.sha256(question.casefold().encode()).hexdigest()
            assert digest == item['question_sha256'] and digest not in seen
            seen.add(digest)
            solution = re.sub(r'<<[^<>]*>>', '', row['answer']).strip()
            expected, status = final_answer(solution)
            assert expected is not None
            # Keep the same human rationale; standardize only the final marker.
            solution = solution.rsplit('####',1)[0].rstrip() + '\n\\boxed{'+str(expected)+'}'
            if cfg.get('native_kind')=='ouro':
                # Native Thinking template already opens <think>; supervise its close.
                reasoning=solution.rsplit('\\boxed{',1)[0].rstrip()
                solution=reasoning+'\n</think>\n\\boxed{'+str(expected)+'}'
            prefix = prompt(tokenizer, question)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            ids = tokenizer.encode(prefix+solution, add_special_tokens=False)+[eos]
            assert ids[:len(prefix_ids)] == prefix_ids, 'Chat/answer token boundary changed'
            assert len(ids) <= cfg['max_sequence_tokens'], f'{split} index{item["index"]}: {len(ids)} tokens; never silently truncate/reselect'
            labels = [-100]*(len(prefix_ids)-1)+ids[len(prefix_ids):]
            assert len(labels) == len(ids)-1 and labels[-1] == eos
            enc[split].append((ids[:-1],labels))
            row.update(prompt=prefix, solution=solution, expected=str(expected), index=item['index'], question_sha256=digest)
            rows[split].append(row); lengths[split].append(len(ids))
    (output/'split_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'tokenization-audit.json').write_text(json.dumps({'protocol':cfg.get('protocol',VERSION),'system':SYSTEM,
        'max_lengths':{k:max(v) for k,v in lengths.items()},'counts':{k:len(v) for k,v in lengths.items()},
        'prompt_masked':True,'eos_scored':True,'truncated_or_reselected':0},indent=2)+'\n')
    return enc,rows,eos


