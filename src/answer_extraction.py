"""GSM extraction v2: last explicit box wins, with balanced braces.

Inspired by MATH/verl's last-box extraction. Unlike verl's DAPO strict mode,
we inspect the whole response rather than an arbitrary final100 characters.
GSM numeric equivalence only; no symbolic evaluation or arbitrary last-number
fallback. Active v1 training runs must be regraded offline, not hot-patched.
"""
import re
from fractions import Fraction
from qwen_protocol import number

VERSION = 'qwen-gsm-last-box-v2'


def last_box(text):
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if not matches:
        return None, 'no_box'
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i], 'last_box'
    return None, 'unclosed_last_box'


def numeric_content(content):
    value = content.strip()
    # Harmless math/typography wrappers, not unit stripping or equation solving.
    for _ in range(8):
        old = value
        if value.startswith('$') and value.endswith('$') and len(value)>1:
            value = value[1:-1].strip()
        wrapped = re.fullmatch(r'\\(?:text|mathrm|textbf)\{(.*)\}', value, flags=re.S)
        if wrapped:
            value = wrapped.group(1).strip()
        if value == old:
            break
    parsed = number(value)
    if parsed is not None:
        return parsed
    fraction = re.fullmatch(r'\\(?:dfrac|tfrac|frac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}',value)
    if fraction is None:
        fraction = re.fullmatch(r'([+-]?\d+)\s*/\s*([+-]?\d+)',value)
    if fraction is not None and int(fraction.group(2)) != 0:
        return Fraction(int(fraction.group(1)),int(fraction.group(2)))
    return None


def final_answer(text):
    content, status = last_box(text)
    if status == 'unclosed_last_box':
        return None, status  # Never silently fall back to an earlier valid box.
    if status == 'last_box':
        parsed = numeric_content(content)
        return parsed, 'last_box' if parsed is not None else 'invalid_last_box'
    if '####' in text:
        value = text.rsplit('####',1)[1].strip()
        parsed = number(value)
        if parsed is None and value.endswith('.') and not value.endswith('..'):
            parsed = number(value[:-1])
        return parsed, 'gsm_marker' if parsed is not None else 'invalid_gsm_marker'
    return None, 'missing_explicit_answer'


def grade(text, expected, eos):
    parsed, status = final_answer(text)
    reference = number(str(expected))
    assert reference is not None
    numerical = parsed is not None and parsed == reference
    return {'parsed_answer': str(parsed) if parsed is not None else None,
            'parse_status': status, 'malformed': parsed is None,
            'numerically_correct': numerical, 'correct': numerical and eos,
            'eos': eos, 'truncated': not eos, 'extraction_protocol': VERSION}
