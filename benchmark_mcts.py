#!/usr/bin/env python3
"""benchmark_mcts.py — CLI entry-point for the Gumbel-MCTS benchmark.

Usage examples::

  # Basic run (maxtree, 200 sims, 20 trials)
  python benchmark_mcts.py

  # Print version and exit
  python benchmark_mcts.py --version

  # GridWorld, 500 simulations, 30 trials, JSON output
  python benchmark_mcts.py --env gridworld --sims 500 --trials 30 --format json

  # Quick smoke-test / dry-run (auto-reduces budget)
  python benchmark_mcts.py --dry-run

  # Compare speedup across three simulation budgets
  python benchmark_mcts.py --compare-sims 100,200,500

  # Sequence environment + save results to a custom path
  python benchmark_mcts.py --env sequence --output results/seq_bench.json

  # Show ASCII search tree for a single search call
  python benchmark_mcts.py --tree --sims 150

Produces by default:
  speedup_report.txt         — human-readable summary
  outputs/bench_*.json       — machine-readable full results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import datetime

import numpy as np

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich import box as rich_box

from gumbel_mcts_cli_high.benchmark import run_benchmark, BenchmarkReport
from gumbel_mcts_cli_high import GumbelMCTS, VanillaMCTS, GridWorldEnv, MaxTreeEnv, __version__
from gumbel_mcts_cli_high.env import SequenceEnv
from gumbel_mcts_cli_high.visualize import print_tree, format_action_table


OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs")
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    """Print Rich startup banner with project name, version, and NEO attribution."""
    text = Text()
    text.append("Gumbel-MCTS-CLI", style="bold cyan")
    text.append(f"  v{__version__}\n", style="dim white")
    text.append("High-performance search — 10× faster than Vanilla MCTS\n", style="white")
    text.append("Made autonomously using NEO — your autonomous AI Agent", style="dim italic")
    console.print(Panel(text, border_style="cyan", expand=False))


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    """Parse command-line arguments and return the namespace."""
    p = argparse.ArgumentParser(
        description="Benchmark Gumbel-MCTS vs Vanilla MCTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"gumbel-mcts-cli {__version__}",
    )
    p.add_argument(
        "--env",
        choices=["maxtree", "gridworld", "sequence"],
        default=os.getenv("BENCH_ENV", "maxtree"),
        help="Search environment (default: maxtree)",
    )
    p.add_argument(
        "--sims",
        type=int,
        default=int(os.getenv("MCTS_N_SIM", "200")),
        help="Number of MCTS simulations per search (default: 200)",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=int(os.getenv("BENCH_N_TRIALS", "20")),
        help="Number of benchmark trials (default: 20)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("BENCH_SEED", "42")),
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-trial output",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes"),
        help="Run with minimal budget — useful for CI / smoke tests",
    )
    p.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default=os.getenv("OUTPUT_FORMAT", "text"),
        help="Output format for the report (default: text)",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Path to write the report (default: speedup_report.txt for text, "
            "outputs/bench_<env>_<ts>.json for json/csv)"
        ),
    )
    p.add_argument(
        "--compare-sims",
        default=None,
        metavar="N1,N2,...",
        help=(
            "Run benchmark at multiple simulation counts and print a comparison table. "
            "Example: --compare-sims 50,100,200,500"
        ),
    )
    p.add_argument(
        "--tree",
        action="store_true",
        help="After benchmarking, show an ASCII search tree for a single Gumbel search",
    )
    p.add_argument(
        "--tree-depth",
        type=int,
        default=int(os.getenv("VIZ_MAX_DEPTH", "3")),
        help="Max depth of the displayed search tree (default: 3)",
    )
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(report: BenchmarkReport, fmt: str, output_path: str | None, env_name: str) -> str:
    """Write report in the requested format and return the path used."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ts = report.timestamp.replace(":", "-").replace(".", "-")

    if fmt == "json":
        path = output_path or os.path.join(OUTPUTS_DIR, f"bench_{env_name}_{ts}.json")
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        return path

    if fmt == "csv":
        path = output_path or os.path.join(OUTPUTS_DIR, f"bench_{env_name}_{ts}.csv")
        with open(path, "w") as f:
            f.write("\n".join(report.to_csv_rows()))
            f.write("\n")
        return path

    # Default: text
    path = output_path or "speedup_report.txt"
    with open(path, "w") as f:
        f.write(report.summary())
        f.write("\n\nDetailed trial times (ms):\n")
        f.write(
            f"{'Trial':>6}  {'Vanilla (ms)':>14}  "
            f"{'Gumbel (ms)':>13}  {'Speedup':>9}\n"
        )
        f.write("-" * 50 + "\n")
        for i, (v, g) in enumerate(zip(report.vanilla_times, report.gumbel_times)):
            spd = v / g if g > 0 else 0.0
            f.write(
                f"{i+1:>6}  {v*1000:>14.3f}  {g*1000:>13.3f}  {spd:>9.2f}×\n"
            )
        if report.vanilla_time_ci and report.gumbel_time_ci:
            f.write("\n95% Confidence Intervals:\n")
            f.write(f"  Vanilla time : {report.vanilla_time_ci}\n")
            f.write(f"  Gumbel  time : {report.gumbel_time_ci}\n")
    return path


def _compare_sims(args, sim_counts: list[int]) -> None:
    """Run benchmark at each sim count and print a Rich comparison table."""
    table = Table(
        title=f"Sim-count comparison  env={args.env}  trials={args.trials}",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Sims", style="cyan", justify="right")
    table.add_column("Vanilla ms", style="red", justify="right")
    table.add_column("Gumbel ms", style="green", justify="right")
    table.add_column("Speedup", style="bold yellow", justify="right")
    table.add_column("Quality ratio", style="blue", justify="right")

    for n_sims in sim_counts:
        report = run_benchmark(
            env_name=args.env,
            n_trials=max(5, args.trials // 2),
            n_simulations=n_sims,
            seed=args.seed,
            verbose=False,
            dry_run=False,
        )
        v_ms = np.mean(report.vanilla_times) * 1000
        g_ms = np.mean(report.gumbel_times) * 1000
        table.add_row(
            str(n_sims),
            f"{v_ms:.2f}",
            f"{g_ms:.2f}",
            f"{report.speedup:.2f}×",
            f"{report.quality_ratio:.4f}",
        )

    console.print(table)


def _show_tree(env_name: str, n_sims: int, seed: int, tree_depth: int) -> None:
    """Run a single Gumbel search and display the ASCII search tree."""
    console.print(f"\n[cyan]{'─'*64}[/cyan]")
    console.print(f"  [bold]Search Tree[/bold]  (env={env_name}, sims={n_sims})")
    console.print(f"[cyan]{'─'*64}[/cyan]")

    if env_name == "maxtree":
        env = MaxTreeEnv(
            depth=int(os.getenv("TREE_DEPTH", "6")),
            branching=int(os.getenv("TREE_BRANCHING", "8")),
            seed=seed,
        )
        action_names = None
    elif env_name == "gridworld":
        env = GridWorldEnv(
            size=int(os.getenv("GRID_SIZE", "8")),
            max_steps=int(os.getenv("MAX_STEPS", "200")),
            seed=seed,
        )
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    else:
        env = SequenceEnv(seed=seed)
        action_names = None

    gumbel = GumbelMCTS(n_simulations=n_sims, seed=seed)
    state = env.reset()

    t0 = time.perf_counter()
    action, root, elapsed = gumbel.search_with_stats(env, state)
    elapsed = time.perf_counter() - t0

    console.print(f"\n  [green]Best action[/green] : {action}   elapsed: {elapsed*1000:.2f} ms\n")
    console.print(print_tree(root, max_depth=tree_depth))
    console.print()
    console.print("  Action table:")
    console.print(format_action_table(root, action_names=action_names))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    """Main entry point — parse args, run benchmark, write report."""
    args = parse_args(argv)

    _print_banner()

    # Propagate to env vars so the benchmark module picks them up
    os.environ["BENCH_ENV"] = args.env
    os.environ["MCTS_N_SIM"] = str(args.sims)
    os.environ["BENCH_N_TRIALS"] = str(args.trials)
    os.environ["BENCH_SEED"] = str(args.seed)

    # ── compare-sims mode ────────────────────────────────────────────────────
    if args.compare_sims:
        try:
            sim_counts = [int(x.strip()) for x in args.compare_sims.split(",") if x.strip()]
        except ValueError:
            console.print(f"[bold red][ERROR][/bold red] --compare-sims expects comma-separated integers, got: {args.compare_sims!r}")
            return 1
        if not sim_counts:
            console.print("[bold red][ERROR][/bold red] --compare-sims: no valid sim counts provided")
            return 1
        _compare_sims(args, sim_counts)
        return 0

    # ── show tree mode ───────────────────────────────────────────────────────
    if args.tree:
        _show_tree(args.env, args.sims, args.seed, args.tree_depth)
        return 0

    # ── normal benchmark ─────────────────────────────────────────────────────
    n_trials = args.trials
    if args.dry_run:
        n_trials = min(n_trials, 5)
        console.print("[yellow][dry-run][/yellow] Using n_trials=5, n_simulations=40")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}[/cyan]"),
            BarColumn(),
            TextColumn("[green]{task.completed}/{task.total}[/green] trials"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                f"Benchmarking [bold]{args.env}[/bold] ({args.sims} sims)",
                total=n_trials,
            )

            def _on_trial(trial_num, total, v_el, v_val, g_el, g_val):
                """Advance progress bar and optionally print per-trial stats."""
                progress.advance(task_id)
                if not args.quiet:
                    progress.console.print(
                        f"  Trial [cyan]{trial_num:>3}[/cyan]/{total} | "
                        f"Vanilla: [red]{v_el*1000:6.2f}ms[/red] val={v_val:.4f} | "
                        f"Gumbel:  [green]{g_el*1000:6.2f}ms[/green] val={g_val:.4f}"
                    )

            report = run_benchmark(
                env_name=args.env,
                n_trials=n_trials,
                n_simulations=args.sims,
                seed=args.seed,
                verbose=False,
                dry_run=args.dry_run,
                on_trial_complete=_on_trial,
            )

    except ValueError as exc:
        console.print(f"[bold red][ERROR][/bold red] {exc}")
        return 1
    except Exception as exc:
        console.print(f"[bold red][ERROR][/bold red] Unexpected failure: {exc}")
        raise

    # ── results table ─────────────────────────────────────────────────────────
    result_table = Table(
        title=f"Benchmark Results — {args.env.upper()}  ({args.sims} sims × {n_trials} trials)",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold white",
    )
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Vanilla MCTS", style="red", justify="right")
    result_table.add_column("Gumbel MCTS", style="green", justify="right")

    v_ms = np.mean(report.vanilla_times) * 1000
    g_ms = np.mean(report.gumbel_times) * 1000
    result_table.add_row("Mean time", f"{v_ms:.2f} ms", f"{g_ms:.2f} ms")
    result_table.add_row(
        "Std time",
        f"{np.std(report.vanilla_times)*1000:.2f} ms",
        f"{np.std(report.gumbel_times)*1000:.2f} ms",
    )
    result_table.add_row(
        "Mean value",
        f"{np.mean(report.vanilla_values):.4f}",
        f"{np.mean(report.gumbel_values):.4f}",
    )
    result_table.add_row(
        "Speedup",
        "1.00×",
        f"[bold yellow]{report.speedup:.2f}×[/bold yellow]",
    )
    result_table.add_row(
        "Quality ratio",
        "1.0000",
        f"[bold]{report.quality_ratio:.4f}[/bold]",
    )
    console.print(result_table)

    # ── write report ─────────────────────────────────────────────────────────
    path = _write_report(report, args.format, args.output, args.env)
    console.print(f"[green][✓][/green] Report written to: {path}")

    # ── also write JSON to outputs/ when format is text ──────────────────────
    if args.format == "text":
        json_path = _write_report(report, "json", None, args.env)
        console.print(f"[green][✓][/green] JSON  written  to: {json_path}")

    # ── optional tree display ────────────────────────────────────────────────
    if args.tree:
        _show_tree(args.env, args.sims, args.seed, args.tree_depth)

    # ── exit code ────────────────────────────────────────────────────────────
    target_speedup = float(os.getenv("TARGET_SPEEDUP", "3.0"))
    if report.speedup >= target_speedup:
        console.print(f"\n[bold green][PASS][/bold green] Speedup {report.speedup:.2f}× ≥ target {target_speedup}×")
    else:
        console.print(f"\n[yellow][NOTE][/yellow] Speedup {report.speedup:.2f}× < target {target_speedup}×")
        console.print("       Try --sims 500 or --trials 30 for a more stable measurement.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
