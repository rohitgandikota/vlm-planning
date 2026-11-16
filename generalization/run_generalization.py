"""Run every generalization experiment of Appendix G, one after another, on one model.

    python generalization/run_generalization.py --model Qwen/Qwen3-VL-4B-Instruct --num_pairs 100
    python generalization/run_generalization.py --model google/gemma-3-12b-it --num_pairs 100
    python generalization/run_generalization.py --search --model Qwen/Qwen3-VL-2B-Instruct   # also the head searches (slow)

Files 01 to 14 test the Qwen3-VL sizes (2B, 4B, 8B, 32B; file 11 the 32B only) and stop with an
error on other models. Files 15 to 19 run on any model of FAMILY_MODELS, Qwen3-VL-8B included. Files
from 20 on test other images and prompts (Appendix G.3). Each file in generalization/main_experiments/
is one experiment and can be run on its own.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # the repository root

from utils.experiment import run_folder

if __name__ == "__main__":
    run_folder(Path(__file__).parent)
