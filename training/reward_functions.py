"""
Reward functions for GRPO training on TrajToM.

Implements R_fmt / R_outcome / R_structure / R_conclusion / R_len
and assembles them into R_total.

Note: tau (trust weight) is currently hardcoded to 1.0.

Design choice: single-function interface `reward_total(...)`.
GRPOTrainer calls this once per rollout batch; internal logging exposes
individual components.
"""

import re
import numpy as np
from typing import List, Optional, Dict


# ============================================================
# Regex patterns
# ============================================================

FMT_PATTERN = re.compile(
    r'<think>.*?</think>\s*<answer>\s*([A-D])\s*</answer>',
    re.DOTALL | re.IGNORECASE,
)

ANSWER_PATTERN = re.compile(
    r'<answer>\s*([A-D])\s*</answer>',
    re.IGNORECASE,
)

CONCLUSION_PATTERN = re.compile(
    r'\[Conclusion\]\s*(.*?)(?=\n\s*\[|</think>|$)',
    re.DOTALL | re.IGNORECASE,
)

REQUIRED_TAGS = ('[Evidence]', '[Inference]', '[Conclusion]')

CLEAN_TRAILING = re.compile(
    r'\s*(this|that|which)\s+(matches|corresponds\s+to|is|supports)\s+option\s+[A-D]\.?\s*$',
    re.IGNORECASE,
)
CLEAN_INLINE = re.compile(r'\boption\s+[A-D]\b', re.IGNORECASE)


# ============================================================
# R_fmt: format compliance (hard constraint, 0 or 1)
# ============================================================

def r_fmt(completion: str) -> float:
    """Return 1.0 if completion matches <think>...</think><answer>X</answer>, else 0.0."""
    return 1.0 if FMT_PATTERN.search(completion) else 0.0


# ============================================================
# R_outcome: answer correctness (0 or 1)
# ============================================================

def extract_answer(completion: str) -> Optional[str]:
    """Extract the answer letter (A-D) from <answer> tag. Returns None if absent."""
    m = ANSWER_PATTERN.search(completion)
    return m.group(1).upper() if m else None


def r_outcome(completion: str, correct_answer: str) -> float:
    """Return 1.0 if extracted answer matches correct_answer, else 0.0."""
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    return 1.0 if pred == correct_answer.upper() else 0.0


# ============================================================
# R_structure: tag completeness (0 to 1, 3-tag template)
# ============================================================

def r_structure(completion: str) -> float:
    """Fraction of required tags present in completion (out of 3)."""
    present = sum(1 for tag in REQUIRED_TAGS if tag in completion)
    return present / len(REQUIRED_TAGS)


# ============================================================
# R_conclusion: semantic alignment with Sketch-based target
# ============================================================

def extract_conclusion(completion: str) -> str:
    """Extract the last [Conclusion] segment. Returns '' if missing or empty."""
    matches = CONCLUSION_PATTERN.findall(completion)
    if not matches:
        return ''
    return matches[-1].strip()


def clean_conclusion(text: str) -> str:
    """Strip 'This matches option X' trailing and inline 'option X' mentions."""
    if not text:
        return ''
    text = CLEAN_TRAILING.sub('', text)
    text = CLEAN_INLINE.sub('', text)
    return text.strip()


def extract_and_clean_conclusion(completion: str) -> str:
    return clean_conclusion(extract_conclusion(completion))


def sim_to_score(sim: float, threshold: float = 0.65) -> float:
    """Map cosine similarity to R_conclusion via piecewise function.

    sim >= threshold -> 1.0
    0 <= sim < threshold -> sim / threshold (linear)
    sim < 0 -> 0 (clipped)
    """
    s = max(0.0, min(1.0, sim))
    return min(s / threshold, 1.0)


def r_conclusion_batch(
    completions: List[str],
    target_embeddings: np.ndarray,
    embedding_model,
    threshold: float = 0.65,
) -> List[float]:
    """Compute R_conclusion for a batch of completions.

    Args:
        completions: N model output strings.
        target_embeddings: np.ndarray shape (N, D), already L2-normalized.
        embedding_model: sentence-transformers model (e.g., all-MiniLM-L6-v2).
        threshold: similarity threshold for full score.

    Returns:
        List of N floats in [0, 1].
    """
    if len(completions) != len(target_embeddings):
        raise ValueError(
            f'completions ({len(completions)}) and target_embeddings '
            f'({len(target_embeddings)}) length mismatch'
        )

    conclusions = [extract_and_clean_conclusion(c) for c in completions]

    # Encode non-empty conclusions only; empty ones score 0
    non_empty_idx = [i for i, c in enumerate(conclusions) if c]
    scores = [0.0] * len(completions)

    if not non_empty_idx:
        return scores

    to_encode = [conclusions[i] for i in non_empty_idx]
    conc_emb = embedding_model.encode(
        to_encode,
        batch_size=max(len(to_encode), 1),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # Both sides L2-normalized -> cos sim = dot product
    for k, i in enumerate(non_empty_idx):
        sim = float(np.dot(conc_emb[k], target_embeddings[i]))
        scores[i] = sim_to_score(sim, threshold)

    return scores


# ============================================================
# R_len: length window + repetition penalty
# ============================================================

def _sigmoid(x: float) -> float:
    # Clamp to avoid overflow
    if x < -50:
        return 0.0
    if x > 50:
        return 1.0
    return 1.0 / (1.0 + np.exp(-x))


def ngram_repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of repeated n-grams (word-level). Returns 0 for short texts."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - (unique / len(ngrams))


def r_rep(text: str, n: int = 4) -> float:
    """Repetition reward: 1 if rep_rate <= 0.1, else exponential decay."""
    rho = ngram_repetition_rate(text, n)
    if rho <= 0.1:
        return 1.0
    return float(np.exp(-8 * (rho - 0.1)))


def r_win(n_tokens: int, low: int = 200, high: int = 2500, width: float = 50.0) -> float:
    """Length window reward: sigmoid ramp at low and high, product in between."""
    lo = _sigmoid((n_tokens - low) / width)
    hi = _sigmoid((high - n_tokens) / width)
    return lo * hi


def r_len(completion: str, tokenizer=None) -> float:
    """Combined length reward: R_rep * R_win.

    tokenizer: optional HF tokenizer for accurate token count.
               If None, falls back to whitespace word count.
    """
    if tokenizer is not None:
        n = len(tokenizer.encode(completion, add_special_tokens=False))
    else:
        n = len(completion.split())
    return r_rep(completion) * r_win(n)


# ============================================================
# R_total: assemble all components
#
# R_total = R_fmt * (R_outcome + tau * R_process) * R_len
# R_process = w_conc * R_conclusion + w_struct * R_structure
#
# NOTE: tau is currently hardcoded to 1.0. After the end-to-end pipeline
# runs on the server, re-enable tau per EmotionThinker formula.
# ============================================================

def r_total_batch(
    completions: List[str],
    correct_answers: List[str],
    target_embeddings: np.ndarray,
    embedding_model,
    tokenizer=None,
    w_conc: float = 0.7,
    w_struct: float = 0.3,
    threshold: float = 0.65,
    tau: float = 1.0,
    num_generations: Optional[int] = None,
    return_components: bool = False,
) -> List[float] or Dict:
    """Compute R_total for a batch of completions.

    Args:
        completions: N model outputs.
        correct_answers: N correct answer letters (A/B/C/D).
        target_embeddings: (N, D) L2-normalized alignment embeddings.
        embedding_model: sentence-transformers model.
        tokenizer: optional HF tokenizer for token-level length.
        w_conc, w_struct: weights for R_process components (sum to 1).
        threshold: R_conclusion similarity threshold.
        tau: fallback scalar trust weight when num_generations is None.
        num_generations: if set, enable group-wise tau. Samples are assumed
            to be laid out in consecutive groups of `num_generations`
            rollouts per prompt (trl GRPOTrainer convention). Each group
            gets tau = mean(outcome) within the group.
        return_components: if True, return dict of all components for logging.

    Returns:
        List of N R_total floats, or dict of component arrays if return_components.
    """
    n = len(completions)
    assert len(correct_answers) == n
    assert len(target_embeddings) == n

    fmt = [r_fmt(c) for c in completions]
    outcome = [r_outcome(c, a) for c, a in zip(completions, correct_answers)]
    structure = [r_structure(c) for c in completions]
    conclusion = r_conclusion_batch(completions, target_embeddings, embedding_model, threshold)
    length = [r_len(c, tokenizer) for c in completions]

    process = [w_conc * conclusion[i] + w_struct * structure[i] for i in range(n)]

    # Group-wise tau: per-prompt answer-correctness ratio
    if num_generations is not None and num_generations > 1:
        if n % num_generations != 0:
            raise ValueError(
                f'batch size ({n}) not divisible by num_generations ({num_generations})'
            )
        tau_per_sample = [0.0] * n
        for g in range(n // num_generations):
            lo = g * num_generations
            hi = lo + num_generations
            tau_g = sum(outcome[lo:hi]) / num_generations
            for i in range(lo, hi):
                tau_per_sample[i] = tau_g
    else:
        tau_per_sample = [tau] * n

    total = [fmt[i] * (outcome[i] + tau_per_sample[i] * process[i]) * length[i] for i in range(n)]

    if return_components:
        return {
            'r_fmt':        np.asarray(fmt, dtype=np.float32),
            'r_outcome':    np.asarray(outcome, dtype=np.float32),
            'r_structure':  np.asarray(structure, dtype=np.float32),
            'r_conclusion': np.asarray(conclusion, dtype=np.float32),
            'r_len':        np.asarray(length, dtype=np.float32),
            'r_process':    np.asarray(process, dtype=np.float32),
            'tau':          np.asarray(tau_per_sample, dtype=np.float32),
            'r_total':      np.asarray(total, dtype=np.float32),
        }
    return total

