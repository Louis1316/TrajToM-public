"""
SFT Data Generation for TrajToM
Converts TrajToM questions' explanation field to structured 4-tag reasoning format.

Features:
- Concurrent processing with configurable workers
- Resume from existing output (--append)
- Graceful shutdown on Ctrl+C
"""

import json
import os
import argparse
import signal
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Global flag for graceful shutdown
shutdown_flag = False
client = None
model_name = None


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_llm(prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.95,
        max_tokens=2_000,
    )
    return response.choices[0].message.content or ''

def signal_handler(sig, frame):
    global shutdown_flag
    print("\n\nReceived Ctrl+C, finishing current tasks and saving...")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)


def build_conversion_prompt(question_data, sketch_phase):
    """Build prompt for converting explanation to 4-tag format."""
    q = question_data

    prompt = f"""Convert the following question's explanation into a structured reasoning format in English.

Input:
- Question: {q['question']}
- Options: {json.dumps(q['options'], ensure_ascii=False)}
- Correct Answer: {q['correct_answer']}
- Explanation: {q['explanation']}

Output format:
<think>
[Evidence] {{Cite specific dialogue using message IDs, e.g., "In [mid X], character says/does..."}}
[Inference] {{Explain how the evidence supports the answer}}
[Conclusion] {{State the final reasoning conclusion that matches the correct option}}
</think>
<answer>{q['correct_answer']}</answer>

Requirements:
1. Evidence must reference specific dialogue using [mid X] format
2. Inference explains the logical connection between evidence and answer
3. Conclusion clearly states the reasoning result
4. Keep it concise, 100-300 words total
5. Output ONLY the formatted result in English, no extra explanation"""

    return prompt


def validate_format(output):
    """Validate output has required 3 tags."""
    required_tags = ['[Evidence]', '[Inference]', '[Conclusion]']
    has_think = '<think>' in output and '</think>' in output
    has_answer = '<answer>' in output and '</answer>' in output
    has_tags = all(tag in output for tag in required_tags)
    return has_think and has_answer and has_tags


def sample_by_subtype(questions, n_per_type=2):
    """Sample n questions per subtype for testing."""
    by_subtype = defaultdict(list)
    for q in questions:
        subtype = q.get('question_subtype', 'unknown')
        by_subtype[subtype].append(q)

    sampled = []
    for subtype, qs in by_subtype.items():
        sampled.extend(qs[:n_per_type])

    print(f"Sampled from {len(by_subtype)} subtypes")
    return sampled


def process_question(q, dialogue_map):
    """Process a single question."""
    global shutdown_flag
    if shutdown_flag:
        return None

    dialogue = dialogue_map.get(q['dialogue_id'], '')
    prompt = build_conversion_prompt(q, q.get('sketch_phase', 'unknown'))

    try:
        if shutdown_flag:
            return None

        output = run_llm(prompt)

        if shutdown_flag:
            return None

        if validate_format(output):
            return {
                'question_id': q['question_id'],
                'dialogue_id': q['dialogue_id'],
                'sketch_id': q['sketch_id'],
                'dialogue': dialogue,
                'question': q['question'],
                'options': q['options'],
                'correct_answer': q['correct_answer'],
                'formatted_reasoning': output
            }
        else:
            return {'failed': q['question_id'], 'reason': 'format_invalid'}
    except Exception as e:
        if shutdown_flag:
            return None
        return {'failed': q['question_id'], 'reason': str(e)}


def generate_sft_data(questions_path, dialogues_path, split_path, output_path, test_mode=False, append=False, workers=4):
    """Generate SFT training data from TrajToM questions."""
    global shutdown_flag

    questions = load_json(questions_path)
    dialogues = load_json(dialogues_path)
    split_data = load_json(split_path)

    dialogue_map = {d['dialogue_id']: d['messages'] for d in dialogues}
    train_dialogue_ids = set(split_data['train'])
    train_questions = [q for q in questions if q['dialogue_id'] in train_dialogue_ids]

    # Load existing data if append mode
    existing_ids = set()
    sft_data = []
    if append and Path(output_path).exists():
        existing_data = load_json(output_path)
        sft_data = existing_data
        existing_ids = {item['question_id'] for item in existing_data}
        print(f"Append mode: loaded {len(existing_ids)} existing samples")

    # Filter out already processed questions
    train_questions = [q for q in train_questions if q['question_id'] not in existing_ids]

    if test_mode:
        train_questions = sample_by_subtype(train_questions, n_per_type=2)
        print(f"Test mode: sampling {len(train_questions)} questions")
    else:
        print(f"Full mode: processing {len(train_questions)} questions")

    if not train_questions:
        print("No questions to process")
        return

    failed = []

    # Concurrent processing
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_question, q, dialogue_map): q for q in train_questions}

        with tqdm(total=len(train_questions), desc="Generating SFT data") as pbar:
            for future in as_completed(futures):
                if shutdown_flag:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                result = future.result()
                if result:
                    if 'failed' in result:
                        failed.append(result)
                    else:
                        sft_data.append(result)
                pbar.update(1)

    # Save results
    save_json(sft_data, output_path)
    print(f"\nGenerated {len(sft_data)} SFT samples (total)")
    print(f"Failed: {len(failed)} samples")

    if failed:
        failed_path = Path(output_path).parent / "sft_failed.json"
        save_json(failed, failed_path)
        print(f"Failed IDs saved to {failed_path}")

    if shutdown_flag:
        print("Interrupted by user, progress saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--test", action="store_true", help="Test mode: sample 2 per subtype")
    parser.add_argument("-a", "--append", action="store_true", help="Append mode: skip existing samples")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of concurrent workers (default: 4)")
    parser.add_argument("--questions", type=str, default=None, help="Path to questions JSON file")
    parser.add_argument("--dialogues", type=str, default=None, help="Path to dialogues JSON file")
    parser.add_argument("--split", type=str, default=None, help="Path to data split JSON file")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON file")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        help="OpenAI-compatible model name")
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL"),
                        help="Optional OpenAI-compatible API base URL")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY before generating SFT rationales.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    model_name = args.model
    base_dir = Path(__file__).resolve().parent.parent / "data"
    artifact_dir = Path(__file__).resolve().parent / "artifacts"

    generate_sft_data(
        questions_path=args.questions or base_dir / "tom_questions.json",
        dialogues_path=args.dialogues or base_dir / "tom_dialogues.json",
        split_path=args.split or base_dir / "data_split.json",
        output_path=args.output or artifact_dir / "sft_train_data.json",
        test_mode=args.test,
        append=args.append,
        workers=args.workers
    )




