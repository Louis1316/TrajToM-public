"""
Theory of Mind (ToM) Questions Evaluation Script
Supports both local models and API calls
Provides detailed accuracy metrics across multiple dimensions
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ModelInterface:
    """Base class for model interfaces"""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class LocalModelInterface(ModelInterface):
    """Interface for local models (using transformers)"""

    def __init__(self, model_path: str, device: str = "cuda"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")

        print(f"Loading local model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        self.device = device
        self.model_name = os.path.basename(model_path.rstrip('/\\'))
        print("Model loaded successfully!")

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        import torch
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        return response

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 512, temperature: float = 0.7) -> List[str]:
        import torch
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            texts.append(text)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        responses = []
        for output in outputs:
            generated_ids = output[prompt_len:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            responses.append(response)
        return responses


class VLLMModelInterface(ModelInterface):
    """Interface for local models using vLLM for fast multi-GPU inference"""

    def __init__(self, model_path: str, tensor_parallel_size: int = 4, max_model_len: int = 4096):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Please install vllm: pip install vllm")

        print(f"Loading model with vLLM from {model_path}...")
        print(f"tensor_parallel_size={tensor_parallel_size}, max_model_len={max_model_len}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="float16",
            trust_remote_code=True,
            max_model_len=max_model_len,
        )
        self.model_name = os.path.basename(model_path.rstrip('/\\'))
        print("Model loaded successfully!")

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        return self.generate_batch([prompt], max_new_tokens, temperature)[0]

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 512, temperature: float = 0.7) -> List[str]:
        from vllm import SamplingParams
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        try:
            outputs = self.llm.chat(
                messages_list,
                sampling_params=sampling_params,
                chat_template_kwargs={"enable_thinking": False},
            )
        except Exception:
            outputs = self.llm.chat(messages_list, sampling_params=sampling_params)
        responses = []
        for output in outputs:
            response = output.outputs[0].text.strip()
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            responses.append(response)
        return responses


class APIModelInterface(ModelInterface):
    """Interface for API-based models (OpenAI, Anthropic, etc.)"""

    DEFAULT_API_KEY = ""  # set via --api-key or OPENAI_API_KEY / ANTHROPIC_API_KEY environment variable
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_type: str = "openai", api_key: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo", base_url: Optional[str] = None):
        self.api_type = api_type.lower()
        self.api_key = api_key or self.DEFAULT_API_KEY or os.getenv(f"{api_type.upper()}_API_KEY")
        self.model_name = model_name
        self.base_url = base_url or self.DEFAULT_BASE_URL

        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            raise ValueError(f"Please set your API key in the code (DEFAULT_API_KEY) or pass api_key parameter")

        if self.api_type == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Please install openai: pip install openai")

            if self.base_url:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                self.client = OpenAI(api_key=self.api_key)

        elif self.api_type == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
            self.client = Anthropic(api_key=self.api_key)

        else:
            raise ValueError(f"Unsupported API type: {api_type}")

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        if self.api_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content

        elif self.api_type == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.content[0].text


class ToMEvaluator:
    """Evaluator for Theory of Mind questions"""

    def __init__(self, model: ModelInterface, data_path: str, dialogue_path: Optional[str] = None,
                 use_cot: bool = False):
        self.model = model
        self.data_path = data_path
        self.dialogue_path = dialogue_path
        self.use_cot = use_cot
        self.questions = self._load_questions()
        self.dialogues = self._load_dialogues()
        self.results = []
        self.errors = []
        self.lock = threading.Lock()  # For thread-safe operations

    def _load_questions(self) -> List[Dict]:
        """Load questions from JSON file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions from {self.data_path}")
        return data

    def _load_dialogues(self) -> Dict[str, Dict]:
        """Load dialogues from JSON file and index by dialogue_id"""
        if not self.dialogue_path:
            # Try to infer dialogue path from data path
            data_dir = os.path.dirname(self.data_path)
            self.dialogue_path = os.path.join(data_dir, "tom_dialogues.json")

        if not os.path.exists(self.dialogue_path):
            print(f"Warning: Dialogue file not found at {self.dialogue_path}")
            print("Questions will be evaluated without dialogue context!")
            return {}

        with open(self.dialogue_path, 'r', encoding='utf-8') as f:
            dialogue_list = json.load(f)

        # Index dialogues by dialogue_id for quick lookup
        dialogues_dict = {d['dialogue_id']: d for d in dialogue_list}
        print(f"Loaded {len(dialogues_dict)} dialogues from {self.dialogue_path}")
        return dialogues_dict

    def _create_prompt(self, question_data: Dict) -> str:
        """Create a prompt for the model"""
        dialogue_id = question_data.get("dialogue_id")
        dialogue_text = ""

        if dialogue_id and dialogue_id in self.dialogues:
            dialogue = self.dialogues[dialogue_id]
            messages = dialogue.get("messages", [])
            dialogue_lines = []
            for msg in messages:
                speaker = msg.get("speaker", "Unknown")
                text = msg.get("text", "")
                dialogue_lines.append(f"{speaker}: {text}")
            dialogue_text = "\n".join(dialogue_lines)
        else:
            dialogue_text = "[No dialogue context available]"

        options_text = "\n".join([f"{k}. {v}" for k, v in question_data["options"].items()])

        if self.use_cot:
            prompt = f"""Read the following dialogue, then answer the multiple-choice question.

DIALOGUE:
{dialogue_text}

---

QUESTION: {question_data["question"]}

OPTIONS:
{options_text}

Let's think step by step. First analyze the dialogue and reason about the answer, then conclude with your final choice in the format: "Answer: X" where X is one of A, B, C, or D."""
        else:
            prompt = f"""Read the following dialogue, then answer the multiple-choice question.

DIALOGUE:
{dialogue_text}

---

QUESTION: {question_data["question"]}

OPTIONS:
{options_text}

Please respond with ONLY the letter of your chosen answer (A, B, C, or D), without any explanation."""

        return prompt

    def _extract_answer(self, response: str, valid_letters: List[str] = None) -> Optional[str]:
        """Extract answer letter from model response (single-choice only)."""
        import re

        if valid_letters is None:
            valid_letters = ['A', 'B', 'C', 'D']

        # Strip <think>...</think> block (SFT / thinking models)
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # <answer>X</answer> tag (SFT model format)
        m = re.search(r'<answer>\s*([A-D])\s*</answer>', clean, re.IGNORECASE)
        if m and m.group(1).upper() in valid_letters:
            return m.group(1).upper()

        # "Answer: X" pattern (CoT format)
        m = re.search(r'(?:Answer|answer|ANSWER)\s*[:：]?\s*\*?\*?\s*([A-D])\b', clean)
        if m and m.group(1).upper() in valid_letters:
            return m.group(1).upper()

        # "The answer is X" / "choice is X" / "option is X" patterns
        m = re.search(r'(?:answer|choice|option)\s+is\s+\*?\*?\s*([A-D])\b', clean, re.IGNORECASE)
        if m and m.group(1).upper() in valid_letters:
            return m.group(1).upper()

        # Last single letter at end of response (fallback for CoT)
        tail = clean[-200:] if len(clean) > 200 else clean
        m = re.search(r'\b([A-D])\b\s*\.?\s*$', tail)
        if m and m.group(1).upper() in valid_letters:
            return m.group(1).upper()

        # Original heuristic on cleaned response (base / instruct models)
        upper = clean.strip().upper()
        for char in valid_letters:
            if char in upper:
                if upper.startswith(char) or f" {char} " in upper or upper == char:
                    return char

        if upper and upper[0] in valid_letters:
            return upper[0]

        return None

    def evaluate_single(self, question_data: Dict) -> Dict:
        """Evaluate a single question"""
        prompt = self._create_prompt(question_data)
        valid_letters = list(question_data.get("options", {}).keys())

        try:
            response = self.model.generate(prompt)
            predicted_answer = self._extract_answer(response, valid_letters)
            correct_answer = question_data["correct_answer"]
            is_correct = predicted_answer == correct_answer

            result = {
                "question_id":        question_data["question_id"],
                "question_type":      question_data["question_type"],
                "question_subtype":   question_data["question_subtype"],
                "reasoning_order":    question_data.get("reasoning_order", "unknown"),
                "sketch_phase":       question_data.get("sketch_phase"),        # order=1 only
                "order2_perspective": question_data.get("order2_perspective"),  # order=2 only
                "correct_answer":     correct_answer,
                "predicted_answer":   predicted_answer,
                "is_correct":         is_correct,
                "raw_response":       response,
                "question":           question_data["question"],
            }

            if not is_correct:
                with self.lock:
                    self.errors.append(result)

            return result

        except Exception as e:
            print(f"Error evaluating question {question_data['question_id']}: {e}")
            error_result = {
                "question_id":        question_data["question_id"],
                "question_type":      question_data["question_type"],
                "question_subtype":   question_data["question_subtype"],
                "reasoning_order":    question_data.get("reasoning_order", "unknown"),
                "sketch_phase":       question_data.get("sketch_phase"),
                "order2_perspective": question_data.get("order2_perspective"),
                "correct_answer":     question_data["correct_answer"],
                "predicted_answer":   None,
                "is_correct":         False,
                "raw_response":       f"ERROR: {str(e)}",
                "question":           question_data["question"],
            }
            with self.lock:
                self.errors.append(error_result)
            return error_result

    def evaluate_all(self, max_questions: Optional[int] = None, concurrency: int = 1, batch_size: int = 1,
                     output_dir: str = "results", resume_from: Optional[str] = None,
                     max_new_tokens: int = 512):
        """Evaluate all questions. Use batch_size>1 for local models, concurrency>1 for API models."""
        questions_to_eval = self.questions[:max_questions] if max_questions else self.questions

        # Set up run folder and handle resume
        if resume_from:
            self._run_folder = resume_from
            checkpoint_file = os.path.join(self._run_folder, "results.json")
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                done_ids = {r["question_id"] for r in self.results}
                self.errors = [r for r in self.results if not r["is_correct"]]
                questions_to_eval = [q for q in questions_to_eval if q["question_id"] not in done_ids]
                print(f"Resuming from {resume_from}: {len(done_ids)} done, {len(questions_to_eval)} remaining.")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = getattr(self.model, 'model_name', 'unknown_model')
            clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            self._run_folder = os.path.join(output_dir, f"{clean_model_name}_{timestamp}")
            os.makedirs(self._run_folder, exist_ok=True)
            checkpoint_file = os.path.join(self._run_folder, "results.json")

        total = len(questions_to_eval)

        use_batch = batch_size > 1 and hasattr(self.model, 'generate_batch')
        if use_batch:
            print(f"\nEvaluating {total} questions with batch_size={batch_size}...")
        else:
            print(f"\nEvaluating {total} questions with concurrency={concurrency}...")
        print(f"Results checkpoint: {checkpoint_file}")

        def _save_checkpoint():
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False)

        if use_batch:
            with tqdm(total=total, desc="Evaluating", unit="q") as pbar:
                for i in range(0, total, batch_size):
                    batch_questions = questions_to_eval[i:i + batch_size]
                    prompts = [self._create_prompt(q) for q in batch_questions]
                    try:
                        responses = self.model.generate_batch(prompts, max_new_tokens=max_new_tokens)
                    except Exception as e:
                        if "out of memory" in str(e).lower():
                            _save_checkpoint()
                            is_vllm = hasattr(self.model, 'llm')
                            if is_vllm:
                                raise RuntimeError(
                                    f"CUDA OOM with vLLM (batch_size={len(batch_questions)}). "
                                    f"Checkpoint saved to: {checkpoint_file}\n"
                                    f"Resume with: --resume {self._run_folder} --max-model-len <smaller_value>"
                                ) from e
                            else:
                                raise RuntimeError(
                                    f"CUDA OOM with batch_size={len(batch_questions)}. "
                                    f"Checkpoint saved to: {checkpoint_file}\n"
                                    f"Resume with: --resume {self._run_folder} --batch-size <smaller_value>"
                                ) from e
                        responses = [f"ERROR: {e}"] * len(batch_questions)

                    for question_data, response in zip(batch_questions, responses):
                        valid_letters = list(question_data.get("options", {}).keys())
                        predicted_answer = self._extract_answer(response, valid_letters)
                        correct_answer = question_data["correct_answer"]
                        is_correct = predicted_answer == correct_answer
                        result = {
                            "question_id":        question_data["question_id"],
                            "question_type":      question_data["question_type"],
                            "question_subtype":   question_data["question_subtype"],
                            "reasoning_order":    question_data.get("reasoning_order", "unknown"),
                            "sketch_phase":       question_data.get("sketch_phase"),
                            "order2_perspective": question_data.get("order2_perspective"),
                            "correct_answer":     correct_answer,
                            "predicted_answer":   predicted_answer,
                            "is_correct":         is_correct,
                            "raw_response":       response,
                            "question":           question_data["question"],
                        }
                        self.results.append(result)
                        if not is_correct:
                            self.errors.append(result)
                        correct = sum(1 for r in self.results if r["is_correct"])
                        pbar.set_postfix(accuracy=f"{correct/len(self.results)*100:.1f}%")
                        pbar.update(1)
                    # Save after every batch
                    _save_checkpoint()

        elif concurrency == 1:
            with tqdm(total=total, desc="Evaluating", unit="q") as pbar:
                for question_data in questions_to_eval:
                    result = self.evaluate_single(question_data)
                    self.results.append(result)
                    correct = sum(1 for r in self.results if r["is_correct"])
                    pbar.set_postfix(accuracy=f"{correct/len(self.results)*100:.1f}%")
                    pbar.update(1)
                    _save_checkpoint()
        else:
            # Concurrent execution (for API models)
            with tqdm(total=total, desc="Evaluating", unit="q") as pbar:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_to_question = {
                        executor.submit(self.evaluate_single, q): q
                        for q in questions_to_eval
                    }
                    for future in as_completed(future_to_question):
                        result = future.result()
                        with self.lock:
                            self.results.append(result)
                            correct = sum(1 for r in self.results if r["is_correct"])
                            pbar.set_postfix(accuracy=f"{correct/len(self.results)*100:.1f}%")
                            pbar.update(1)
                            _save_checkpoint()

        print(f"\nEvaluation complete! Processed {total} questions.")

    def _calculate_accuracy(self, results: List[Dict]) -> float:
        """Calculate accuracy from results"""
        if not results:
            return 0.0
        correct = sum(1 for r in results if r["is_correct"])
        return correct / len(results) * 100

    def generate_report(self) -> Dict:
        """Generate detailed accuracy report"""
        report = {
            "total_questions": len(self.results),
            "total_correct":   sum(1 for r in self.results if r["is_correct"]),
            "overall_accuracy": self._calculate_accuracy(self.results),
        }

        def _group(key_fn):
            groups = defaultdict(list)
            for r in self.results:
                groups[key_fn(r)].append(r)
            return {
                k: {"count": len(v), "correct": sum(1 for r in v if r["is_correct"]),
                    "accuracy": self._calculate_accuracy(v)}
                for k, v in sorted(groups.items(), key=lambda x: (x[0] is None, x[0] or ''))
            }

        report["by_question_type"]    = _group(lambda r: r["question_type"])
        report["by_question_subtype"] = _group(lambda r: r["question_subtype"])
        report["by_reasoning_order"]  = _group(lambda r: r["reasoning_order"])

        # sketch_phase: order=1 questions only
        o1 = [r for r in self.results if r.get("reasoning_order") == 1]
        report["by_sketch_phase"] = {
            k: v for k, v in _group(lambda r: r.get("sketch_phase", "unknown")).items()
            if k is not None
        } if o1 else {}

        # order2_perspective: order=2 questions only
        o2 = [r for r in self.results if r.get("reasoning_order") == 2]
        report["by_order2_perspective"] = {
            k: v for k, v in _group(lambda r: r.get("order2_perspective", "unknown")).items()
            if k is not None
        } if o2 else {}

        return report

    def print_report(self):
        """Print formatted report"""
        report = self.generate_report()

        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)

        print(f"\nOverall Statistics:")
        print(f"  Total Questions: {report['total_questions']}")
        print(f"  Correct Answers: {report['total_correct']}")
        print(f"  Overall Accuracy: {report['overall_accuracy']:.2f}%")

        print(f"\nAccuracy by Question Type:")
        for qtype, stats in report["by_question_type"].items():
            print(f"  {qtype}: {stats['correct']}/{stats['count']}  {stats['accuracy']:.1f}%")

        print(f"\nAccuracy by Question Subtype:")
        for subtype, stats in report["by_question_subtype"].items():
            print(f"  {subtype}: {stats['correct']}/{stats['count']}  {stats['accuracy']:.1f}%")

        print(f"\nAccuracy by Reasoning Order:")
        for order, stats in report["by_reasoning_order"].items():
            print(f"  order={order}: {stats['correct']}/{stats['count']}  {stats['accuracy']:.1f}%")

        if report.get("by_sketch_phase"):
            print(f"\nAccuracy by Sketch Phase (order=1):")
            for phase, stats in report["by_sketch_phase"].items():
                print(f"  {phase}: {stats['correct']}/{stats['count']}  {stats['accuracy']:.1f}%")

        if report.get("by_order2_perspective"):
            print(f"\nAccuracy by Perspective (order=2):")
            for persp, stats in report["by_order2_perspective"].items():
                print(f"  {persp}: {stats['correct']}/{stats['count']}  {stats['accuracy']:.1f}%")

        print(f"\nTotal Errors: {len(self.errors)}")
        print("="*80 + "\n")

    def save_results(self, output_dir: str = "results"):
        """Save detailed results and errors to files"""
        # Reuse the folder created in evaluate_all if available
        run_folder = getattr(self, '_run_folder', None)
        if run_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = getattr(self.model, 'model_name', 'unknown_model')
            clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            run_folder = os.path.join(output_dir, f"{clean_model_name}_{timestamp}")
            os.makedirs(run_folder, exist_ok=True)

        # Save all results
        results_file = os.path.join(run_folder, "results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {results_file}")

        # Save errors
        errors_file = os.path.join(run_folder, "errors.json")
        with open(errors_file, 'w', encoding='utf-8') as f:
            json.dump(self.errors, f, indent=2, ensure_ascii=False)
        print(f"Errors saved to: {errors_file}")

        # Save report
        report = self.generate_report()
        report_file = os.path.join(run_folder, "report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {report_file}")

        # Save human-readable report
        readable_report_file = os.path.join(run_folder, "report.txt")
        with open(readable_report_file, 'w', encoding='utf-8') as f:
            import sys
            from io import StringIO

            # Capture print output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            self.print_report()
            report_text = sys.stdout.getvalue()
            sys.stdout = old_stdout

            f.write(report_text)
        print(f"Readable report saved to: {readable_report_file}")
        print(f"\nAll results saved in folder: {run_folder}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ToM questions")
    parser.add_argument("--data-path", type=str,
                       default="../data/tom_questions_test.json",
                       help="Path to the questions JSON file")
    parser.add_argument("--dialogue-path", type=str,
                       default="../data/tom_dialogues_test.json",
                       help="Path to the dialogues JSON file (optional, will auto-detect if not specified)")
    parser.add_argument("--model-type", type=str, choices=["local", "vllm", "api"], required=True,
                       help="Type of model to use")

    # Local model arguments
    parser.add_argument("--model-path", type=str, help="Path to local model (for local model type)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device to use for local model")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                       help="Number of GPUs for tensor parallelism (vLLM only, default: 4)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                       help="Max sequence length for vLLM KV cache (default: 4096)")

    # API model arguments
    parser.add_argument("--api-type", type=str, choices=["openai", "anthropic"], default="openai",
                       help="API type (for api model type, default: openai)")
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo",
                       help="Model name for API")

    # Evaluation arguments
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions to evaluate (for testing)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--concurrency", type=int, default=1,
                       help="Number of concurrent requests for API models (default: 1)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for local model inference (default: 1, recommended: 16-32 for A800 80GB)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to a previous run folder to resume from its checkpoint")
    parser.add_argument("--use-cot", action="store_true",
                       help="Use Chain-of-Thought prompt (let model think step by step). Default: direct answer.")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                       help="Max new tokens. Default: 512 (direct) or 1024 (CoT).")

    args = parser.parse_args()

    # Initialize model
    if args.model_type == "local":
        if not args.model_path:
            parser.error("--model-path is required for local model type")
        model = LocalModelInterface(args.model_path, args.device)
    elif args.model_type == "vllm":
        if not args.model_path:
            parser.error("--model-path is required for vllm model type")
        model = VLLMModelInterface(args.model_path, args.tensor_parallel_size, args.max_model_len)
    else:  # api
        if not args.api_type:
            parser.error("--api-type is required for api model type")
        model = APIModelInterface(args.api_type, model_name=args.model_name)

    # Initialize evaluator
    evaluator = ToMEvaluator(model, args.data_path, args.dialogue_path, use_cot=args.use_cot)

    # Decide max_new_tokens (CoT needs more space)
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 1024 if args.use_cot else 512

    # Run evaluation
    evaluator.evaluate_all(args.max_questions, args.concurrency, args.batch_size, args.output_dir,
                           args.resume, max_new_tokens=max_new_tokens)

    # Print and save results
    evaluator.print_report()
    evaluator.save_results(args.output_dir)


if __name__ == "__main__":
    main()
