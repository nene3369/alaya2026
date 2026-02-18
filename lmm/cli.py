"""LMM CLI — command-line interface."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from lmm.core import LMM


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lmm",
        description="LMM — Classical QUBO Optimizer (no D-Wave required)",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of items to select")
    parser.add_argument("--alpha", type=float, default=1.0, help="Surprise weight")
    parser.add_argument("--gamma", type=float, default=10.0, help="Constraint weight")
    parser.add_argument(
        "--method", choices=["sa", "ising_sa", "relaxation", "greedy"], default="sa",
        help="Solver method",
    )
    parser.add_argument("--input", type=str, help="Input .npy file")
    parser.add_argument("--demo", action="store_true", help="Run demo")

    args = parser.parse_args(argv)

    if args.demo:
        _run_demo(args)
    elif args.input:
        _run_file(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_demo(args: argparse.Namespace) -> None:
    print("=== LMM Demo ===\n")
    rng = np.random.default_rng(42)
    n, k = 100, args.k
    surprises = rng.exponential(scale=2.0, size=n)
    print(f"Candidates: {n}")
    print(f"Select: {k}")
    print(f"Method: {args.method}")
    print(f"Surprise range: [{surprises.min():.3f}, {surprises.max():.3f}]\n")

    model = LMM(k=k, alpha=args.alpha, gamma=args.gamma, solver_method=args.method)
    result = model.select_from_surprises(surprises)

    print(f"Selected indices: {result.selected_indices}")
    print(f"Selected surprises: {result.surprise_values}")
    print(f"Total surprise: {result.surprise_values.sum():.3f}")
    print(f"Energy: {result.energy:.3f}")

    topk = np.argsort(surprises)[-k:]
    print("\n--- Comparison: naive Top-K ---")
    print(f"Top-K total surprise: {surprises[topk].sum():.3f}")


def _run_file(args: argparse.Namespace) -> None:
    data = np.load(args.input)
    print(f"Loaded: {args.input} (shape={data.shape})")
    model = LMM(k=args.k, alpha=args.alpha, gamma=args.gamma, solver_method=args.method)
    result = model.select_from_surprises(data)
    print(f"Selected indices: {result.selected_indices}")
    print(f"Energy: {result.energy:.3f}")


if __name__ == "__main__":
    main()
