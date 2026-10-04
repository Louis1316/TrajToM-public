"""
Build alignment targets for R_conclusion computation.

Phase 1.1 of GRPO data preparation pipeline.

For each question in the training set, constructs the 'alignment_target' string
by extracting and combining relevant Sketch fields per Section 2.2.4 of method.md.

Input:
  - tom_questions.json  (question metadata)
  - tom_dialogues.json  (dialogues with profile_metadata for character names)
  - tom_sketches.json   (psychological sketches)
  - data_split.json     (train/dev split)

Output:
  - alignment_targets.jsonl  (one JSON per question with alignment_target string)
"""

import json
import re
import argparse
from pathlib import Path


SUBTYPE_TO_DIM = {
    'emotion_reasoning':   'emotion',
    'belief_reasoning':    'belief',
    'intention_reasoning': 'intention',
    'desire_reasoning':    'desire',
}


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_character_name(dialogue, character):
    """Return name for 'A' or 'B' character."""
    meta = dialogue['profile_metadata']
    return meta['person_a']['name'] if character == 'A' else meta['person_b']['name']


def extract_believed_part(second_order_str):
    """
    Extract the 'X thinks Y [state] Z' part from a second_order_states string.

    Actual format example:
      'Sutton thinks Rowan believes Sutton failed to upload ... (actual: ...) | triggered by: ...'

    We strip the '(actual: ...)' parenthetical block and the '| triggered by: ...' suffix,
    returning the prefix as the believed part.
    """
    if not second_order_str:
        return ''
    s = second_order_str
    # Remove (actual: ...) block
    s = re.sub(r'\s*\([^)]*actual:[^)]*\)', '', s, flags=re.IGNORECASE)
    # Remove | triggered by: ... tail
    s = re.sub(r'\s*\|\s*triggered by:.*$', '', s, flags=re.IGNORECASE | re.DOTALL)
    return s.strip().rstrip('.').strip()


def build_shift_str(shifts_dict):
    """
    Build 'belief: from X to Y; intention: from P to Q' from key_moment.shifts.{char}.
    Only include dimensions whose value is not 'unchanged'.
    """
    if not shifts_dict:
        return 'general stance shift'
    parts = []
    for dim in ('belief', 'intention', 'emotion'):
        val = shifts_dict.get(dim, 'unchanged')
        if val and 'unchanged' not in val.lower():
            parts.append(f'{dim}: {val}')
    return '; '.join(parts) if parts else 'general stance shift'


def build_target_t1_order1(question, sketch):
    """T1 order=1: phase_states.{sketch_phase}.{target_character}.{dim}

    Special case: desire is cross-phase stable, sketch_phase may be None in questions.
    Fall back to 'opening' phase for desire.
    """
    phase = question.get('sketch_phase')
    char = question['target_character']
    dim = SUBTYPE_TO_DIM[question['question_subtype']]
    if phase is None:
        if dim == 'desire':
            phase = 'opening'  # desire is stable across phases
        else:
            raise ValueError(f"sketch_phase is None for non-desire question: {question['question_id']}")
    return sketch['phase_states'][phase][char][dim]


def build_target_t1_order2(question, sketch):
    """T1 order=2: second_order_states.{order2_perspective}.{dim}, extract believed_part.

    Schema allows some second_order fields to be null (B_about_A: ≥2/3 non-null,
    A_about_B: ≥1/2 non-null). Return empty string if the required field is null —
    caller will mark the question as 'null_second_order' and skip it.
    """
    perspective = question['order2_perspective']
    dim = SUBTYPE_TO_DIM[question['question_subtype']]
    raw = sketch['second_order_states'][perspective].get(dim)
    if raw is None:
        return ''
    return extract_believed_part(raw)


def build_target_t2(question, sketch):
    """T2 behavior_explanation: causal_chain.{causal_chain_key}"""
    chain_key = question['causal_chain_key']
    return sketch['causal_chain'][chain_key]


def build_target_t3_km(question, sketch, name):
    """T3 key_moment_identification: 'Turning point: {trigger}. Observable shift in {name}: {shift_str}.'"""
    km = sketch['phase_states']['key_moment']
    trigger = km['trigger']
    char = question['target_character']
    shift_str = build_shift_str(km['shifts'].get(char, {}))
    return f"Turning point: {trigger}. Observable shift in {name}: {shift_str}."


def build_target_t3_cda(question, sketch, name):
    """T3 change_direction_assessment: '{name}'s {dim} shifts from {s_start} to {s_end}.'"""
    char = question['target_character']
    dim = question['cda_dimension']
    # sketch_phase format: '{start}_{end}', e.g. 'opening_closing'
    phase_start, phase_end = question['sketch_phase'].split('_', 1)
    s_start = sketch['phase_states'][phase_start][char][dim]
    s_end = sketch['phase_states'][phase_end][char][dim]
    return f"{name}'s {dim} shifts from '{s_start}' to '{s_end}'."


def build_target_t3_ccr(question, sketch, name):
    """T3 change_cause_reasoning: '{name} shifts because {trigger}, resulting in {shift_str}.'"""
    km = sketch['phase_states']['key_moment']
    trigger = km['trigger']
    char = question['target_character']
    shift_str = build_shift_str(km['shifts'].get(char, {}))
    return f"{name} shifts because {trigger}, resulting in {shift_str}."


def build_alignment_target(question, sketch, dialogue):
    """Dispatch to the correct builder based on question_subtype and reasoning_order."""
    subtype = question['question_subtype']
    order = question.get('reasoning_order', 1)

    # Type 1: mental state reasoning
    if subtype in SUBTYPE_TO_DIM:
        if order == 1:
            return build_target_t1_order1(question, sketch)
        elif order == 2:
            return build_target_t1_order2(question, sketch)

    # Type 2: behavioral reasoning
    if subtype == 'behavior_explanation':
        return build_target_t2(question, sketch)

    # Type 3: trajectory reasoning — needs character name
    char = question.get('target_character')
    name = get_character_name(dialogue, char) if char else ''

    if subtype == 'key_moment_identification':
        return build_target_t3_km(question, sketch, name)
    if subtype == 'change_direction_assessment':
        return build_target_t3_cda(question, sketch, name)
    if subtype == 'change_cause_reasoning':
        return build_target_t3_ccr(question, sketch, name)

    raise ValueError(f"Unknown subtype/order: {subtype} / {order}")


def main(args):
    base = Path(args.data_dir)

    questions = load_json(base / 'tom_questions.json')
    dialogues = load_json(base / 'tom_dialogues.json')
    sketches = load_json(base / 'tom_sketches.json')
    split = load_json(base / 'data_split.json')

    dialogue_map = {d['dialogue_id']: d for d in dialogues}
    sketch_map = {s['scenario_id']: s['sketch'] for s in sketches}

    split_ids = set(split[args.split])
    filtered = [q for q in questions if q['dialogue_id'] in split_ids]
    print(f'Split={args.split}: {len(filtered)} questions')

    out = []
    errors = []

    for q in filtered:
        try:
            dialogue = dialogue_map[q['dialogue_id']]
            sketch = sketch_map[q['sketch_id']]
            target = build_alignment_target(q, sketch, dialogue)
            if not target or not target.strip():
                errors.append({'question_id': q['question_id'], 'reason': 'empty_target'})
                continue
            out.append({
                'question_id': q['question_id'],
                'dialogue_id': q['dialogue_id'],
                'sketch_id': q['sketch_id'],
                'question_subtype': q['question_subtype'],
                'reasoning_order': q.get('reasoning_order', 1),
                'alignment_target': target.strip(),
            })
        except Exception as e:
            errors.append({'question_id': q['question_id'], 'reason': str(e)})

    out_path = Path(args.output) if args.output else base / f'alignment_targets_{args.split}.jsonl'
    save_jsonl(out, out_path)
    print(f'Saved {len(out)} alignment targets to {out_path}')
    print(f'Errors: {len(errors)}')
    if errors:
        err_path = out_path.with_suffix('.errors.json')
        with open(err_path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f'Error details saved to {err_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=str(Path(__file__).resolve().parent.parent / 'data'),
                        help='Directory with tom_questions.json, tom_sketches.json, etc.')
    parser.add_argument('--split', choices=['train', 'dev'], default='train')
    parser.add_argument('--output', default=None,
                        help='Output .jsonl path (default: <data_dir>/alignment_targets_<split>.jsonl)')
    args = parser.parse_args()
    main(args)
