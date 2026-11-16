"""Run every binding experiment of the main paper, one after another.

    python binding/run_binding.py --num_pairs 500            # the paper's held-out pairs
    python binding/run_binding.py --num_pairs 50             # a quick look
    python binding/run_binding.py --appendix                 # also the appendix experiments
    python binding/run_binding.py --search                   # also the head search (slow)

Each file in binding/main_experiments/ is one experiment and can be run on its own.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # the repository root

from utils.experiment import run_folder

if __name__ == "__main__":
    run_folder(Path(__file__).parent)
