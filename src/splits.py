"""Deterministic dataset splitting with exact-input leakage prevention."""
import hashlib
import json
import random


def split_rows(rows, seed=42, eval_fraction=0.1, test_fraction=0.1):
    if not 0 < eval_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("eval and test fractions must be between zero and one")
    if eval_fraction + test_fraction >= 1:
        raise ValueError("eval and test fractions must sum to less than one")
    groups = {}
    for row in rows:
        # Treat whitespace-only changes as duplicates; never split identical inputs.
        key = " ".join(row["input"].split())
        groups.setdefault(key, []).append(row)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("at least three distinct log inputs are required")
    random.Random(seed).shuffle(keys)
    n_test = max(1, round(len(keys) * test_fraction))
    n_eval = max(1, round(len(keys) * eval_fraction))
    if n_test + n_eval >= len(keys):
        raise ValueError("fractions leave no training input groups")
    partitions = (keys[n_test+n_eval:], keys[n_test:n_test+n_eval], keys[:n_test])
    return tuple([row for key in part for row in groups[key]] for part in partitions)


def dataset_fingerprint(rows):
    encoded = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
