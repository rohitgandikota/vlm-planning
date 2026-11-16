"""Run every relay experiment of the main paper, one after another.

    python relay/run_relay.py --num_pairs 500            # the paper's held-out pairs
    python relay/run_relay.py --num_pairs 50             # a quick look
    python relay/run_relay.py --appendix                 # also the appendix experiments
    python relay/run_relay.py --search                   # also the head search (slow)

Each file in relay/main_experiments/ is one experiment and can be run on its own.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # the repository root

from utils.experiment import run_folder

if __name__ == "__main__":
    run_folder(Path(__file__).parent)
