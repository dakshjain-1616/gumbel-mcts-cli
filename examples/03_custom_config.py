#!/usr/bin/env python3
"""03_custom_config.py — Configure behaviour via env vars and constructor params.

Shows how to tune Gumbel-MCTS parameters and all three environments.

Run from any directory:
    python examples/03_custom_config.py

    # Or override settings via env vars:
    GUMBEL_K=8 GUMBEL_C_SCALE=0.3 python examples/03_custom_config.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gumbel_mcts_cli_high import GumbelMCTS, GridWorldEnv, MaxTreeEnv, SequenceEnv

# ── Config A: constructor params ─────────────────────────────────────────────
gumbel_fast = GumbelMCTS(
    n_simulations=80,
    max_considered_actions=8,   # top-k at root (power of 2)
    c_scale=0.3,                # completed-Q interpolation weight
    rollout_depth=20,           # shorter rollouts → faster
    seed=42,
)
gumbel_quality = GumbelMCTS(
    n_simulations=300,
    max_considered_actions=16,
    c_scale=0.5,
    rollout_depth=50,
    seed=42,
)

print("── Config A vs B on MaxTree ─────────────────────────────────")
env = MaxTreeEnv(depth=5, branching=4, seed=99)
state = env.reset()

a_fast, r_fast, t_fast = gumbel_fast.search_with_stats(env, state)
env2 = MaxTreeEnv(depth=5, branching=4, seed=99)
a_qual, r_qual, t_qual = gumbel_quality.search_with_stats(env2, env2.reset())

print(f"  Fast   (80 sims) : action={a_fast}  Q={r_fast.value:.4f}  {t_fast*1000:.2f} ms")
print(f"  Quality(300 sims): action={a_qual}  Q={r_qual.value:.4f}  {t_qual*1000:.2f} ms")
print(f"  Optimal value    : {env.optimal_value:.4f}")

# ── Config B: env var override ────────────────────────────────────────────────
print("\n── Active env-var settings ─────────────────────────────────")
settings = {
    "MCTS_N_SIM":       os.getenv("MCTS_N_SIM", "200"),
    "GUMBEL_K":         os.getenv("GUMBEL_K", "16"),
    "GUMBEL_C_SCALE":   os.getenv("GUMBEL_C_SCALE", "0.5"),
    "ROLLOUT_DEPTH":    os.getenv("ROLLOUT_DEPTH", "50"),
    "BENCH_ENV":        os.getenv("BENCH_ENV", "maxtree"),
    "BENCH_N_TRIALS":   os.getenv("BENCH_N_TRIALS", "20"),
    "BENCH_SEED":       os.getenv("BENCH_SEED", "42"),
}
for k, v in settings.items():
    print(f"  {k:<20} = {v}")

# ── Config C: all three environments ────────────────────────────────────────
print("\n── All three environments ──────────────────────────────────")
configs = [
    ("GridWorld 6×6",  GridWorldEnv(size=6, max_steps=60, seed=1)),
    ("MaxTree d=4 b=4", MaxTreeEnv(depth=4, branching=4, seed=1)),
    ("SequenceEnv v=6 d=4", SequenceEnv(vocab_size=6, depth=4, seed=1)),
]
gumbel = GumbelMCTS(n_simulations=120, seed=7)

for label, env in configs:
    state = env.reset()
    action, root, elapsed = gumbel.search_with_stats(env, state)
    print(f"  {label:<22} → action={action}  Q={root.value:.4f}  {elapsed*1000:.2f} ms")
