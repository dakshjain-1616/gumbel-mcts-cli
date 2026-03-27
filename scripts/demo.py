#!/usr/bin/env python3
"""scripts/demo.py — CLI entry point for the Gumbel-MCTS demo.

Delegates to demo.py in the project root.
"""
import sys
import os

# Ensure the project root is on sys.path so that demo.py and gumbel_mcts are importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Change working directory to project root so relative paths (outputs/) resolve correctly
os.chdir(_project_root)

from demo import main  # noqa: E402

if __name__ == "__main__":
    main()
