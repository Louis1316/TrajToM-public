"""
Precompute embeddings for alignment targets.

Phase 1.1 step 2 of GRPO data preparation pipeline.

Loads alignment_targets_{split}.jsonl, encodes each target with
sentence-transformers/all-MiniLM-L6-v2, and saves:
  - alignment_embeddings_{split}.npy       shape (N, 384), float32, L2-normalized
  - alignment_embeddings_{split}.ids.json  list of question_id matching npy rows

Separating .npy + .ids.json (instead of storing per-row in jsonl) keeps the
training-time reward function fast: it can batch-dot-product against a single
preloaded matrix.
"""

import json
import argparse
from pathlib import Path

import numpy as np


MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def resolve_device(device_arg):
    if device_arg != 'auto':
        return device_arg
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def main(args):
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f'Input file not found: {in_path}')

    items = load_jsonl(in_path)
    targets = [x['alignment_target'] for x in items]
    ids = [x['question_id'] for x in items]
    print(f'Loaded {len(items)} alignment targets from {in_path}')

    device = resolve_device(args.device)
    print(f'Using device: {device}')

    from sentence_transformers import SentenceTransformer
    print(f'Loading model: {MODEL_NAME}')
    model = SentenceTransformer(MODEL_NAME, device=device)

    embeddings = model.encode(
        targets,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    print(f'Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}')

    out_npy = Path(args.output_npy)
    out_ids = Path(args.output_ids)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_ids.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_npy, embeddings)
    with open(out_ids, 'w', encoding='utf-8') as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    print(f'Saved embeddings to {out_npy}')
    print(f'Saved ID index to {out_ids}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='Path to alignment_targets_{split}.jsonl')
    parser.add_argument('--output_npy', required=True,
                        help='Path to output .npy file')
    parser.add_argument('--output_ids', required=True,
                        help='Path to output .ids.json file')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device for encoding (default: auto)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Encoding batch size (default: 128)')
    args = parser.parse_args()
    main(args)
