"""
Convert sft_train_data.json to LLaMA-Factory Alpaca format.
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


def format_input(item):
    """Format dialogue + question as user input."""
    dialogue_lines = "\n".join(
        f"[mid {m['mid']}] {m['speaker']}: {m['text']}"
        for m in item['dialogue']
    )
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(item['options'].items()))

    return f"Dialogue:\n{dialogue_lines}\n\nQuestion: {item['question']}\n\nOptions:\n{options_text}"


def convert(input_path, output_path, split, split_file=None):
    data = json.load(open(input_path, 'r', encoding='utf-8'))

    # Filter by split if split_file provided
    if split_file and split == 'dev':
        split_data = json.load(open(split_file, 'r', encoding='utf-8'))
        dev_ids = set(split_data['dev'])
        data = [d for d in data if d['dialogue_id'] in dev_ids]

    alpaca_data = []
    for item in data:
        alpaca_data.append({
            "instruction": SYSTEM_PROMPT,
            "input": format_input(item),
            "output": item['formatted_reasoning']
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(alpaca_data, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"Saved {len(alpaca_data)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--split_file", default=None)
    args = parser.parse_args()

    convert(args.input, args.output, args.split, args.split_file)
