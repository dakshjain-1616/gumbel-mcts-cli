#!/usr/bin/env python3
"""02_advanced_usage.py — search_with_stats, anytime search, and tree visualizer.

Run from any directory:
    python examples/02_advanced_usage.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gumbel_mcts_cli_high import GumbelMCTS, VanillaMCTS, GridWorldEnv
from gumbel_mcts_cli_high.visualize import print_tree, format_action_table

# ── 1. search_with_stats ─────────────────────────────────────────────────────
env = GridWorldEnv(size=6, max_steps=80, seed=7)
state = env.reset()

vanilla = VanillaMCTS(n_simulations=150, seed=1)
gumbel  = GumbelMCTS(n_simulations=150, seed=1)

va, vroot, velapsed = vanilla.search_with_stats(env, state)
ga, groot, gelapsed = gumbel.search_with_stats(env, state)

action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
print("── search_with_stats ───────────────────────────────────────")
print(f"  Vanilla MCTS : action={action_names[va]:<5}  time={velapsed*1000:.2f} ms  Q={vroot.value:.4f}")
print(f"  Gumbel  MCTS : action={action_names[ga]:<5}  time={gelapsed*1000:.2f} ms  Q={groot.value:.4f}")
print(f"  Speedup      : {velapsed/gelapsed:.2f}×")

# ── 2. Anytime search ────────────────────────────────────────────────────────
print("\n── anytime search (yield after each halving round) ─────────")
env2 = GridWorldEnv(size=6, max_steps=80, seed=7)
state2 = env2.reset()
gumbel2 = GumbelMCTS(n_simulations=200, seed=2)

last_snapshot = None
for snapshot in gumbel2.search_anytime(env2, state2):
    a = snapshot["action"]
    name = action_names.get(a, str(a)) if a is not None else "None"
    print(
        f"  round={snapshot['round']}  candidates={snapshot['n_candidates']}"
        f"  best={name}  elapsed={snapshot['elapsed_sec']*1000:.2f} ms"
    )
    last_snapshot = snapshot

print(f"  Final best action: {action_names.get(last_snapshot['action'], last_snapshot['action'])}")

# ── 3. ASCII tree visualizer ─────────────────────────────────────────────────
print("\n── ASCII search tree (depth=2) ─────────────────────────────")
env3 = GridWorldEnv(size=6, max_steps=80, seed=7)
gumbel3 = GumbelMCTS(n_simulations=150, seed=3)
_, root3, _ = gumbel3.search_with_stats(env3, env3.reset())

print(print_tree(root3, max_depth=2))
print()
print(format_action_table(root3, action_names=action_names))
