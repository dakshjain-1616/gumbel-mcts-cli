#!/usr/bin/env python3
"""01_quick_start.py — Minimal Gumbel-MCTS example.

Run from any directory:
    python examples/01_quick_start.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gumbel_mcts_cli_high import GumbelMCTS, MaxTreeEnv

# Build a MaxTree environment (depth=5, branching=4)
env = MaxTreeEnv(depth=5, branching=4, seed=42)
state = env.reset()

# Create the searcher and run one search call
gumbel = GumbelMCTS(n_simulations=100, seed=0)
best_action = gumbel.search(env, state)

print(f"Best action at root: {best_action}")
print(f"Optimal value      : {env.optimal_value:.4f}")
print("Done.")
