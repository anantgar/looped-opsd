"""Small fixed-question trace tables built from already-generated predictions."""
import json
from pathlib import Path


def trace_rows(path, offset, rows, step, family, per_depth=2):
    references={row['question_sha256']:row for row in rows[:per_depth]}
    table=[]
    with Path(path).open() as stream:
        stream.seek(offset)
        for line in stream:
            prediction=json.loads(line)
            if prediction['family']!=family or prediction['step']!=step:continue
            reference=references.get(prediction['question_sha256'])
            if reference is None:continue
            table.append([step,family,prediction['depth'],prediction['index'],reference['question'],
                reference['solution'],prediction['text'],prediction['generated_tokens'],
                prediction['eos'],prediction['malformed'],prediction['correct']])
    return table


COLUMNS=['step','family','depth','index','question','reference','generated','tokens','eos','malformed','correct']
