#!/usr/bin/env python3
"""04_full_pipeline.py — End-to-end benchmark pipeline.

Runs a full head-to-head benchmark (Vanilla MCTS vs Gumbel MCTS), prints a
summary with 95% confidence intervals, and writes results to outputs/.

Run from any directory:
    python examples/04_full_pipeline.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json

from gumbel_mcts_cli_high.benchmark import run_benchmark
from gumbel_mcts_cli_high.visualize import format_confidence_summary, print_tree, format_action_table
from gumbel_mcts_cli_high import GumbelMCTS, MaxTreeEnv

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Step 1: Run benchmark ────────────────────────────────────────────────────
print("=" * 60)
print("  Gumbel-MCTS Full Pipeline Demo")
print("=" * 60)
print("\nStep 1: Running head-to-head benchmark (dry-run mode)...")

report = run_benchmark(
    env_name="maxtree",
    n_trials=10,
    n_simulations=150,
    seed=42,
    verbose=True,
    dry_run=False,
)

# ── Step 2: Display summary with confidence intervals ───────────────────────
print("\nStep 2: Confidence intervals")
print(format_confidence_summary(report.vanilla_times, label="Vanilla time (s)"))
print(format_confidence_summary(report.gumbel_times,  label="Gumbel  time (s)"))
print(format_confidence_summary(report.vanilla_values, label="Vanilla value  "))
print(format_confidence_summary(report.gumbel_values,  label="Gumbel  value  "))

# ── Step 3: Per-trial breakdown ──────────────────────────────────────────────
print("\nStep 3: Per-trial breakdown")
print(f"  {'Trial':>5}  {'Vanilla ms':>12}  {'Gumbel ms':>11}  {'Speedup':>9}")
print("  " + "-" * 44)
for i, (v, g) in enumerate(zip(report.vanilla_times, report.gumbel_times)):
    spd = v / g if g > 0 else 0.0
    print(f"  {i+1:>5}  {v*1000:>12.3f}  {g*1000:>11.3f}  {spd:>9.2f}×")

# ── Step 4: Write outputs ────────────────────────────────────────────────────
print("\nStep 4: Writing output files...")

json_path = os.path.join(OUTPUTS_DIR, "pipeline_demo.json")
with open(json_path, "w") as f:
    json.dump(report.to_dict(), f, indent=2)
print(f"  JSON  → {json_path}")

csv_path = os.path.join(OUTPUTS_DIR, "pipeline_demo.csv")
with open(csv_path, "w") as f:
    f.write("\n".join(report.to_csv_rows()))
    f.write("\n")
print(f"  CSV   → {csv_path}")

txt_path = os.path.join(OUTPUTS_DIR, "pipeline_demo_summary.txt")
with open(txt_path, "w") as f:
    f.write(report.summary())
    f.write("\n")
print(f"  Text  → {txt_path}")

# ── Step 5: ASCII search tree for a single Gumbel call ───────────────────────
print("\nStep 5: ASCII search tree for a single Gumbel search")
env = MaxTreeEnv(depth=5, branching=4, seed=42)
gumbel = GumbelMCTS(n_simulations=150, seed=0)
action, root, elapsed = gumbel.search_with_stats(env, env.reset())

print(f"  Best action: {action}   elapsed: {elapsed*1000:.2f} ms")
print()
print(print_tree(root, max_depth=2))
print()
print(format_action_table(root))

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  Speedup       : {report.speedup:.2f}×")
print(f"  Quality ratio : {report.quality_ratio:.4f}")
print("=" * 60)
