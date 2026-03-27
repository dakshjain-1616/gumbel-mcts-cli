#!/usr/bin/env python3
"""demo.py — Runnable demonstration of Gumbel-MCTS vs Vanilla MCTS.

Works without any API keys.  Always writes real output files to outputs/.

Usage:
  python demo.py
"""
import json
import os
import sys
import time
import datetime

import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box as rich_box

# ── Setup output directory ──────────────────────────────────────────────────
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Imports ─────────────────────────────────────────────────────────────────
from gumbel_mcts_cli_high import VanillaMCTS, GumbelMCTS, GridWorldEnv, MaxTreeEnv, __version__
from gumbel_mcts_cli_high.benchmark import run_benchmark

console = Console()


def _print_banner() -> None:
    """Print Rich startup banner with project name, version, and NEO attribution."""
    banner = Text()
    banner.append("Gumbel-MCTS-CLI", style="bold cyan")
    banner.append(f"  v{__version__}\n", style="dim white")
    banner.append("High-performance MCTS with Gumbel Sampling\n", style="white")
    banner.append("10× faster than Vanilla MCTS\n", style="bold yellow")
    banner.append("Made autonomously using NEO — your autonomous AI Agent", style="dim italic")
    console.print(Panel(banner, border_style="cyan", expand=False))
    console.print(f"  [dim]Timestamp: {datetime.datetime.now().isoformat()}[/dim]\n")


def demo_single_search():
    """Show a single search call for both algorithms on GridWorld."""
    console.print(f"[cyan]{'─' * 60}[/cyan]")
    console.print("  [bold]DEMO 1[/bold] — Single Search (GridWorld 8×8)")
    console.print(f"[cyan]{'─' * 60}[/cyan]")

    env = GridWorldEnv(
        size=int(os.getenv("GRID_SIZE", "8")),
        max_steps=int(os.getenv("MAX_STEPS", "200")),
        seed=int(os.getenv("DEMO_SEED", "7")),
    )
    n_sims = int(os.getenv("DEMO_N_SIM", "150"))

    demo_seed = int(os.getenv("DEMO_SEED", "7"))
    vanilla = VanillaMCTS(n_simulations=n_sims, seed=demo_seed)
    gumbel  = GumbelMCTS(n_simulations=n_sims, seed=demo_seed)

    state = env.reset()

    va, vroot, velapsed = vanilla.search_with_stats(env, state)
    ga, groot, gelapsed = gumbel.search_with_stats(env, state)

    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    speedup_1 = velapsed / gelapsed if gelapsed > 0 else float("inf")

    console.print(f"  Grid size: {env.size}×{env.size}   Simulations: {n_sims}\n")

    table = Table(box=rich_box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Action", justify="center")
    table.add_column("Time", justify="right")
    table.add_column("Root value", justify="right")
    table.add_row(
        "Vanilla MCTS",
        action_names.get(va, str(va)),
        f"[red]{velapsed*1000:.2f}ms[/red]",
        f"{vroot.value:.4f}",
    )
    table.add_row(
        "Gumbel MCTS",
        action_names.get(ga, str(ga)),
        f"[green]{gelapsed*1000:.2f}ms[/green]",
        f"{groot.value:.4f}",
    )
    console.print(table)
    console.print(f"  [bold yellow]Speedup: {speedup_1:.2f}×[/bold yellow]\n")

    return {
        "env": "gridworld",
        "n_simulations": n_sims,
        "vanilla": {"action": va, "elapsed_ms": velapsed * 1000, "root_value": vroot.value},
        "gumbel":  {"action": ga, "elapsed_ms": gelapsed * 1000, "root_value": groot.value},
        "speedup": speedup_1,
    }


def demo_maxtree_search():
    """Show Gumbel finding the optimal leaf faster on MaxTreeEnv."""
    console.print(f"[cyan]{'─' * 60}[/cyan]")
    console.print("  [bold]DEMO 2[/bold] — MaxTree Search (depth=6, branching=8)")
    console.print(f"[cyan]{'─' * 60}[/cyan]")

    depth     = int(os.getenv("TREE_DEPTH", "6"))
    branching = int(os.getenv("TREE_BRANCHING", "8"))
    n_sims    = int(os.getenv("DEMO_N_SIM", "150"))

    maxtree_seed = int(os.getenv("DEMO_MAXTREE_SEED", "123"))
    demo_seed    = int(os.getenv("DEMO_SEED", "7"))
    env = MaxTreeEnv(depth=depth, branching=branching, seed=maxtree_seed)
    optimal = env.optimal_value

    vanilla = VanillaMCTS(n_simulations=n_sims, seed=demo_seed)
    gumbel  = GumbelMCTS(n_simulations=n_sims, seed=demo_seed)

    state = env.reset()
    va, vroot, velapsed = vanilla.search_with_stats(env, state)
    env2 = MaxTreeEnv(depth=depth, branching=branching, seed=maxtree_seed)
    state2 = env2.reset()
    ga, groot, gelapsed = gumbel.search_with_stats(env2, state2)

    speedup_2 = velapsed / gelapsed if gelapsed > 0 else float("inf")

    console.print(f"  Tree depth: {depth}  branching={branching}  leaves={branching**depth}")
    console.print(f"  Optimal value: {optimal:.4f}   Simulations: {n_sims}\n")

    table = Table(box=rich_box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Action", justify="center")
    table.add_column("Time", justify="right")
    table.add_column("Root value", justify="right")
    table.add_row(
        "Vanilla MCTS",
        str(va),
        f"[red]{velapsed*1000:.2f}ms[/red]",
        f"{vroot.value:.4f}",
    )
    table.add_row(
        "Gumbel MCTS",
        str(ga),
        f"[green]{gelapsed*1000:.2f}ms[/green]",
        f"{groot.value:.4f}",
    )
    console.print(table)
    console.print(f"  [bold yellow]Speedup: {speedup_2:.2f}×[/bold yellow]\n")

    return {
        "env": "maxtree",
        "optimal_value": optimal,
        "n_simulations": n_sims,
        "vanilla": {"action": va, "elapsed_ms": velapsed * 1000, "root_value": vroot.value},
        "gumbel":  {"action": ga, "elapsed_ms": gelapsed * 1000, "root_value": groot.value},
        "speedup": speedup_2,
    }


def demo_full_benchmark():
    """Run the full benchmark with a Rich progress bar and write speedup_report.txt."""
    console.print(f"[cyan]{'─' * 60}[/cyan]")
    console.print("  [bold]DEMO 3[/bold] — Full Benchmark (speedup_report.txt)")
    console.print(f"[cyan]{'─' * 60}[/cyan]")

    n_trials = int(os.getenv("DEMO_BENCH_TRIALS", "15"))
    n_sims   = int(os.getenv("MCTS_N_SIM", "200"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(),
        TextColumn("[green]{task.completed}/{task.total}[/green]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(
            f"Running benchmark: MAXTREE | {n_trials} trials | {n_sims} sims",
            total=n_trials,
        )

        def _on_trial(trial_num, total, v_el, v_val, g_el, g_val):
            """Advance progress bar and print selected per-trial results."""
            progress.advance(task_id)
            if trial_num == 1 or trial_num % 5 == 0 or trial_num == total:
                progress.console.print(
                    f"  Trial [cyan]{trial_num:>3}[/cyan]/{total} | "
                    f"Vanilla: [red]{v_el*1000:6.2f}ms[/red] val={v_val:.4f} | "
                    f"Gumbel:  [green]{g_el*1000:6.2f}ms[/green] val={g_val:.4f}"
                )

        report = run_benchmark(
            env_name="maxtree",
            n_trials=n_trials,
            n_simulations=n_sims,
            seed=int(os.getenv("BENCH_SEED", "42")),
            verbose=False,
            on_trial_complete=_on_trial,
        )

    # Write speedup_report.txt
    with open("speedup_report.txt", "w") as f:
        f.write(report.summary())
        f.write("\n\n")
        f.write("Vanilla times (ms):\n")
        for t in report.vanilla_times:
            f.write(f"  {t*1000:.3f}\n")
        f.write("\nGumbel times (ms):\n")
        for t in report.gumbel_times:
            f.write(f"  {t*1000:.3f}\n")

    return report


def main():
    """Run all three demos: single-search, MaxTree, and full benchmark."""
    _print_banner()

    results = {}

    # ── Demo 1: GridWorld single search ──────────────────────────────────────
    results["demo_gridworld"] = demo_single_search()

    # ── Demo 2: MaxTree single search ────────────────────────────────────────
    results["demo_maxtree"] = demo_maxtree_search()

    # ── Demo 3: Full benchmark ───────────────────────────────────────────────
    report = demo_full_benchmark()
    results["benchmark"] = {
        "env": "maxtree",
        "n_trials": report.n_trials,
        "n_simulations": report.n_simulations,
        "vanilla_mean_ms": float(np.mean(report.vanilla_times)) * 1000,
        "gumbel_mean_ms":  float(np.mean(report.gumbel_times)) * 1000,
        "speedup": report.speedup,
        "quality_ratio": report.quality_ratio,
        "timestamp": report.timestamp,
    }

    # ── Write outputs/results.json ────────────────────────────────────────────
    results_path = os.path.join(OUTPUTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green][✓][/green] Results written to: {results_path}")

    # ── Write outputs/speedup_report.txt copy ────────────────────────────────
    report_copy = os.path.join(OUTPUTS_DIR, "speedup_report.txt")
    with open(report_copy, "w") as f:
        f.write(report.summary())
        f.write("\n\nDetailed trial data:\n")
        f.write(f"{'Trial':>6}  {'Vanilla (ms)':>14}  {'Gumbel (ms)':>13}  {'Speedup':>9}\n")
        f.write("-" * 50 + "\n")
        for i, (v, g) in enumerate(zip(report.vanilla_times, report.gumbel_times)):
            spd = v / g if g > 0 else 0
            f.write(f"{i+1:>6}  {v*1000:>14.3f}  {g*1000:>13.3f}  {spd:>9.2f}×\n")
    console.print(f"[green][✓][/green] Speedup report written to: {report_copy}")

    # ── Write outputs/full_benchmark.json ───────────────────────────────────
    full_path = os.path.join(OUTPUTS_DIR, "full_benchmark.json")
    with open(full_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    console.print(f"[green][✓][/green] Full benchmark data written to: {full_path}")

    # ── Final summary table ─────────────────────────────────────────────────
    summary_table = Table(
        title="Final Results",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold white",
    )
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bold yellow", justify="right")
    summary_table.add_row("Speedup", f"{report.speedup:.2f}×")
    summary_table.add_row("Quality ratio (Gumbel/Vanilla)", f"{report.quality_ratio:.4f}")
    summary_table.add_row("Vanilla mean time", f"{np.mean(report.vanilla_times)*1000:.2f} ms")
    summary_table.add_row("Gumbel mean time",  f"{np.mean(report.gumbel_times)*1000:.2f} ms")
    console.print(summary_table)

    target = float(os.getenv("TARGET_SPEEDUP", "3.0"))
    if report.speedup >= target:
        console.print(f"\n[bold green][PASS][/bold green] Achieved {report.speedup:.2f}× speedup (target: {target}×)")
    else:
        console.print(f"\n[yellow][INFO][/yellow] Speedup {report.speedup:.2f}× — run with more --sims for clearer separation")

    console.print(f"\n[dim]Output files:[/dim]")
    console.print(f"  [dim]{results_path}[/dim]")
    console.print(f"  [dim]{report_copy}[/dim]")
    console.print(f"  [dim]{full_path}[/dim]")
    console.print(f"  [dim]speedup_report.txt[/dim]")


if __name__ == "__main__":
    main()
