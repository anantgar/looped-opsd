"""Qwen study protocol v1; explicit final answers, never arbitrary last numbers."""
import re
from decimal import Decimal, InvalidOperation

SYSTEM = 'Solve the math problem step by step. Put your final numerical answer in \\boxed{...}.'
VERSION = 'qwen-gsm-explicit-answer-v1'


def number(value):
    value = value.strip().removeprefix('$').strip()
    if not re.fullmatch(r'[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)', value):
        return None
    try:
        return Decimal(value.replace(',', ''))
    except InvalidOperation:
        return None


def final_answer(text):
    """Accept numeric #### suffixes or explicit boxes; reject conflicting answers.

    Deliberately excludes prose conclusions and intermediate unmarked numbers.
    Multiple identical boxes are allowed; nonnumeric boxes count as malformed.
    """
    candidates = []
    if '####' in text:
        candidates.append(number(text.rsplit('####', 1)[1].strip().rstrip('.')))
    boxes = re.findall(r'\\boxed\{([^{}]*)\}', text)
    candidates.extend(number(value) for value in boxes)
    if not candidates or any(value is None for value in candidates):
        return None, 'missing_or_invalid_explicit_answer'
    if len(set(candidates)) != 1:
        return None, 'conflicting_explicit_answers'
    return candidates[0], 'explicit_answer'


def prompt(tokenizer, question):
    return tokenizer.apply_chat_template([
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': question.strip()},
    ], tokenize=False, add_generation_prompt=True)


def grade(text, expected, eos):
    value, status = final_answer(text)
    expected_value = number(str(expected))
    assert expected_value is not None
    numerical = value is not None and value == expected_value
    return {'parsed_answer': str(value) if value is not None else None,
            'parse_status': status, 'malformed': value is None,
            'numerically_correct': numerical, 'correct': numerical and eos,
            'eos': eos, 'truncated': not eos}
