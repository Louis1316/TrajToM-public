"""
GRPO training script for TrajToM.

TRL GRPOTrainer with Sketch-grounded process reward.

Usage (server):
  python train_grpo.py \
    --base_model Qwen/Qwen3.5-9B-Instruct \
    --sft_adapter_path /path/to/sft-lora \
    --train_data ./artifacts/grpo_train.jsonl \
    --alignment_npy ./artifacts/alignment_embeddings_train.npy \
    --alignment_ids ./artifacts/alignment_embeddings_train.ids.json \
    --output_dir ./saves/trajtom-grpo-pilot \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.85
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from sentence_transformers import SentenceTransformer

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from reward_functions import r_total_batch


EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'  # set to local path if needed


def build_reward_fn(
    embedding_model: SentenceTransformer,
    embedding_matrix: np.ndarray,  # (N_train, 384)
    tokenizer,
    w_conc: float = 0.7,
    w_struct: float = 0.3,
    threshold: float = 0.65,
    num_generations: int = 8,
):
    """Build a TRL-compatible reward callable via closure.

    TRL GRPOTrainer calls reward functions with signature:
        reward_fn(prompts, completions, **kwargs)
    where kwargs contain per-sample columns from the dataset.

    We rely on 'embedding_row' and 'correct_answer' being dataset columns.
    """

    def reward_fn(prompts, completions, **kwargs):
        # TRL passes kwargs as parallel lists (one per sample)
        embedding_rows = kwargs['embedding_row']
        correct_answers = kwargs['correct_answer']

        # Look up pre-computed alignment embeddings for this batch
        target_embeddings = embedding_matrix[embedding_rows]  # (B, 384)

        components = r_total_batch(
            completions=completions,
            correct_answers=correct_answers,
            target_embeddings=target_embeddings,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            w_conc=w_conc,
            w_struct=w_struct,
            threshold=threshold,
            num_generations=num_generations,
            return_components=True,
        )

        # TRL expects a list of scalar rewards
        return components['r_total'].tolist()

    return reward_fn


def load_model_with_adapter(base_model_path: str, sft_adapter_path: str):
    """Load base model and merge SFT LoRA adapter (kept trainable for GRPO)."""
    import os
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print(f'Loading base model: {base_model_path}')
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=f'cuda:{local_rank}',
        trust_remote_code=True,
    )

    print(f'Loading SFT LoRA adapter: {sft_adapter_path}')
    model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main(args):
    # -------- Data --------
    print(f'Loading training data from {args.train_data}')
    train_ds = load_dataset('json', data_files=args.train_data, split='train')
    print(f'Train set: {len(train_ds)} samples')

    # Optional dev set
    eval_ds = None
    if args.dev_data and Path(args.dev_data).exists():
        print(f'Loading dev data from {args.dev_data}')
        eval_ds = load_dataset('json', data_files=args.dev_data, split='train')
        print(f'Dev set: {len(eval_ds)} samples')

    # -------- Alignment embeddings --------
    print(f'Loading alignment embeddings: {args.alignment_npy}')
    embedding_matrix = np.load(args.alignment_npy).astype(np.float32)
    print(f'Embedding matrix shape: {embedding_matrix.shape}')

    # -------- Embedding model (CPU, small) --------
    print(f'Loading embedding model: {EMBEDDING_MODEL}')
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')

    # -------- Policy model (GPU) --------
    model, tokenizer = load_model_with_adapter(args.base_model, args.sft_adapter_path)

    # -------- Build reward function --------
    reward_fn = build_reward_fn(
        embedding_model=embedding_model,
        embedding_matrix=embedding_matrix,
        tokenizer=tokenizer,
        w_conc=args.w_conc,
        w_struct=args.w_struct,
        threshold=args.threshold,
        num_generations=args.num_generations,
    )

    # -------- GRPO config --------
    grpo_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        beta=args.kl_beta,  # KL coefficient
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy='no',
        bf16=True,
        report_to='tensorboard',
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
    )
    if args.vllm_mode == 'colocate':
        grpo_kwargs['vllm_gpu_memory_utilization'] = args.vllm_gpu_memory_utilization
    else:
        # server mode: HTTP client -> external vllm-serve process
        grpo_kwargs['vllm_server_host'] = args.vllm_server_host
        grpo_kwargs['vllm_server_port'] = args.vllm_server_port

    grpo_config = GRPOConfig(**grpo_kwargs)

    # -------- Trainer --------
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # vLLM 0.17.1 loads Qwen3.5-9B as Qwen3_5ForConditionalGeneration (multimodal),
    # so parameter names need "language_model." prefix to match vLLM's internal names.
    if args.use_vllm and hasattr(trainer, 'vllm_generation'):
        _orig_fix = trainer.vllm_generation._fix_param_name_to_vllm
        def _patched_fix(name, extra_prefixes=None):
            name = _orig_fix(name, extra_prefixes)
            return 'language_model.' + name
        trainer.vllm_generation._fix_param_name_to_vllm = _patched_fix

    print('Starting training...')
    trainer.train()

    print(f'Saving final model to {args.output_dir}')
    trainer.save_model(args.output_dir)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--base_model', default='Qwen/Qwen3.5-9B-Instruct')
    parser.add_argument('--sft_adapter_path', required=True,
                        help='Path to SFT LoRA adapter directory')

    # Data
    parser.add_argument('--train_data', required=True,
                        help='Path to grpo_pilot.jsonl or grpo_train.jsonl')
    parser.add_argument('--dev_data', default=None,
                        help='Optional path to grpo_dev.jsonl')
    parser.add_argument('--alignment_npy', required=True,
                        help='Path to alignment_embeddings_*.npy')
    parser.add_argument('--alignment_ids', default=None,
                        help='Path to alignment_embeddings_*.ids.json (for sanity check)')

    # Output
    parser.add_argument('--output_dir', required=True)

    # Training hyperparameters (Section 3.4 of method.md)
    parser.add_argument('--num_epochs', type=float, default=1.0)
    parser.add_argument('--per_device_batch_size', type=int, default=1)
    parser.add_argument('--grad_accum_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--kl_beta', type=float, default=0.001)
    parser.add_argument('--num_generations', type=int, default=8,
                        help='N rollouts per prompt for GRPO')
    parser.add_argument('--max_prompt_length', type=int, default=2048)
    parser.add_argument('--max_completion_length', type=int, default=1024)

    # Reward function hyperparameters (Section 2.2.4)
    parser.add_argument('--w_conc', type=float, default=0.7)
    parser.add_argument('--w_struct', type=float, default=0.3)
    parser.add_argument('--threshold', type=float, default=0.65)

    # Logging and eval
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=100)

    # vLLM rollout acceleration
    parser.add_argument('--use_vllm', action='store_true')
    parser.add_argument('--vllm_mode', type=str, default='colocate',
                        choices=['colocate', 'server'],
                        help='colocate = vLLM on same GPUs as training (slower). '
                             'server = external vllm-serve on dedicated GPU (faster).')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.6)
    parser.add_argument('--vllm_server_host', type=str, default='localhost')
    parser.add_argument('--vllm_server_port', type=int, default=8000)

    args = parser.parse_args()
    main(args)



