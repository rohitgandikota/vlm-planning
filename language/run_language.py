"""Run every language experiment of the main paper, one after another.

    python language/run_language.py --num_pairs 500               # the paper's held-out pairs
    python language/run_language.py --num_pairs 50                # a quick look
    python language/run_language.py --coco /data/coco             # where COCO val2014 is, for the COCO experiment
    python language/run_language.py --appendix                    # also the appendix experiments
    python language/run_language.py --search                      # also the head search (slow)

Each file in language/main_experiments/ is one experiment and can be run on its own.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # the repository root

from utils.experiment import run_folder

if __name__ == "__main__":
    if "--coco" in sys.argv:                 # only the COCO experiment needs it: hand it over as a variable
        at = sys.argv.index("--coco")
        os.environ["COCO_FOLDER"] = sys.argv[at + 1]
        del sys.argv[at:at + 2]
    run_folder(Path(__file__).parent)
