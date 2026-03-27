# Examples

Runnable scripts demonstrating Gumbel-MCTS-CLI features. Each script adds
`sys.path.insert` so it works from any directory.

```bash
python examples/01_quick_start.py
python examples/02_advanced_usage.py
python examples/03_custom_config.py
python examples/04_full_pipeline.py
```

## Scripts

| Script | What it demonstrates |
|---|---|
| [`01_quick_start.py`](01_quick_start.py) | Minimal working example — create a `MaxTreeEnv`, run `GumbelMCTS.search()`, print the best action (~15 lines) |
| [`02_advanced_usage.py`](02_advanced_usage.py) | `search_with_stats()` timing, `search_anytime()` generator with per-round snapshots, ASCII tree + action table visualizer |
| [`03_custom_config.py`](03_custom_config.py) | Constructor-level tuning (`n_simulations`, `max_considered_actions`, `c_scale`, `rollout_depth`), active env-var listing, all three environments (`GridWorldEnv`, `MaxTreeEnv`, `SequenceEnv`) |
| [`04_full_pipeline.py`](04_full_pipeline.py) | End-to-end benchmark pipeline: `run_benchmark()`, 95% confidence intervals, per-trial table, JSON/CSV/text output to `outputs/`, ASCII tree for a single search call |
