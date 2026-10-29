"""Predeclared midway diagnostic on matching generated train/validation subsets."""
import math

def midway_check(events,depth,step):
    def metrics(family,s):
        matches=[e['metrics'][depth] for e in events if e['event']==family and e['step']==s]
        if len(matches)!=1:raise ValueError(f'Missing/duplicate midway diagnostic: {family} step{s}')
        return matches[0]
    vi,vm=metrics('validation_generation',0),metrics('validation_generation',step)
    ti,tm=metrics('train_generation',0),metrics('train_generation',step)
    assert vi['examples']==vm['examples'] and ti['examples']==tm['examples']
    drop=vi['accuracy']-vm['accuracy'];gain=tm['accuracy']-ti['accuracy']
    gap_growth=(tm['accuracy']-vm['accuracy'])-(ti['accuracy']-vi['accuracy'])
    # Avoid acting on one or two changed answers. This is a diagnostic, not significance testing.
    overfit=((drop>=.25 and gain>=.125 and vi['correct']-vm['correct']>=4)
             or (gain>=.25 and drop>=0 and gap_growth>=.25 and tm['correct']-ti['correct']>=4))
    return {'step':step,'overfit':overfit,'initial_validation':vi,'midway_validation':vm,
            'initial_train':ti,'midway_train':tm,'validation_accuracy_drop':drop,
            'train_accuracy_gain':gain,'train_validation_gap_growth':gap_growth,
            'decision':'stop full-weight and start fresh core-only' if overfit else 'finish full-weight then run fresh core-only',
            'small_fixed_subset_diagnostic':True}
