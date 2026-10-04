"""
Build final GRPO training dataset.

Phase 1.1 step 3 of GRPO data preparation pipeline.

Combines:
  - tom_questions.json       (metadata + correct answer)
  - tom_dialogues.json       (dialogue messages)
  - data_split.json          (train/dev split)
  - alignment_targets_{split}.jsonl        (target text, from step 1)
  - alignment_embeddings_{split}.ids.json  (row index into npy, from step 2)

Only keeps questions that have a valid alignment_target (some T1 order=2
questions are skipped because the second_order_states field is null per schema).

Output:
  grpo_{split}.jsonl  with fields:
    - question_id
    - prompt              (system + dialogue + question + options, SFT-style)
    - correct_answer
    - question_subtype
    - reasoning_order
    - alignment_target
    - sketch_id
    - embedding_row       (row index into alignment_embeddings_{split}.npy)
"""

import json
import argparse
from pathlib import Path


SYSTEM_PROMPT = """You are an expert in Theory of Mind reasoning. Given a dialogue and a multiple-choice question about a character's mental state, reason step by step and select the correct answer.

Your response must follow this format:
<think>
[Evidence] Cite specific dialogue turns using [mid X] format.
[Inference] Explain how the evidence supports your conclusion.
[Conclusion] State your final reasoning conclusion.
</think>
<answer>X</answer>"""


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def render_prompt(dialogue, question):
    """Render the user-facing prompt: dialogue + question + options (SFT-style)."""
    dialogue_lines = '\n'.join(
        f"[mid {m['mid']}] {m['speaker']}: {m['text']}"
        for m in dialogue['messages']
    )
    options_text = '\n'.join(f"{k}. {v}" for k, v in sorted(question['options'].items()))

    user_input = (
        f"Dialogue:\n{dialogue_lines}\n\n"
        f"Question: {question['question']}\n\n"
        f"Options:\n{options_text}"
    )

    return f"<system>\n{SYSTEM_PROMPT}\n</system>\n\n<user>\n{user_input}\n</user>"


def main(args):
    base = Path(args.data_dir)

    questions = load_json(base / 'tom_questions.json')
    dialogues = load_json(base / 'tom_dialogues.json')
    split = load_json(base / 'data_split.json')

    targets = load_jsonl(base / f'alignment_targets_{args.split}.jsonl')
    ids_json = load_json(base / f'alignment_embeddings_{args.split}.ids.json')

    dialogue_map = {d['dialogue_id']: d for d in dialogues}
    target_map = {t['question_id']: t['alignment_target'] for t in targets}
    embedding_row_map = {qid: row for row, qid in enumerate(ids_json)}

    split_ids = set(split[args.split])
    filtered = [q for q in questions if q['dialogue_id'] in split_ids]
    print(f'Split={args.split}: {len(filtered)} questions in split')

    out = []
    skipped_no_target = 0
    skipped_no_embedding = 0

    for q in filtered:
        qid = q['question_id']
        if qid not in target_map:
            skipped_no_target += 1
            continue
        if qid not in embedding_row_map:
            skipped_no_embedding += 1
            continue

        dialogue = dialogue_map[q['dialogue_id']]
        prompt = render_prompt(dialogue, q)

        out.append({
            'question_id': qid,
            'prompt': prompt,
            'correct_answer': q['correct_answer'],
            'question_subtype': q['question_subtype'],
            'reasoning_order': q.get('reasoning_order', 1),
            'alignment_target': target_map[qid],
            'sketch_id': q['sketch_id'],
            'embedding_row': embedding_row_map[qid],
        })

    out_path = Path(args.output) if args.output else base / f'grpo_{args.split}.jsonl'
    save_jsonl(out, out_path)

    print(f'Saved {len(out)} GRPO samples to {out_path}')
    print(f'Skipped (no alignment_target): {skipped_no_target}')
    print(f'Skipped (no embedding row):    {skipped_no_embedding}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=str(Path(__file__).resolve().parent.parent / 'data'))
    parser.add_argument('--split', choices=['train', 'dev'], default='train')
    parser.add_argument('--output', default=None,
                        help='Output .jsonl path (default: <data_dir>/grpo_<split>.jsonl)')
    args = parser.parse_args()
    main(args)
