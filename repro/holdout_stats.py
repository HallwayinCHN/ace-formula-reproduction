"""Predeclared paired, question-cluster bootstrap and fixed-price accounting."""

import numpy as np


def paired_interval(left, right, seed=2026, repetitions=10000):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("Expected equally shaped question-by-three-seed matrices")
    differences = (left - right).mean(axis=1)
    indices = np.random.default_rng(seed).integers(0, len(differences), size=(repetitions, len(differences)))
    distribution = differences[indices].mean(axis=1) * 100
    return {"delta_percentage_points": float(differences.mean() * 100),
            "ci95_percentage_points": np.quantile(distribution, [0.025, 0.975]).tolist(),
            "bootstrap_seed": seed, "repetitions": repetitions, "clusters": len(differences),
            "unit": "question; all three trained-book seeds kept together"}


def standardized_cost(prompt_tokens, output_tokens, config):
    return ((prompt_tokens * config["input_usd_per_million"]
             + output_tokens * config["output_usd_per_million"]) / 1_000_000
            * config["usd_to_cny_budget_factor"])
