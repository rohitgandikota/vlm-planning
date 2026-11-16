"""Gaze heads read the panel that the pointer refers to (Section 6.1).

We copy the queries of the 32 gaze heads at the colon token from the source run into the target
run. If these heads use the pointer to look back at the image, the target should now answer with
its own letter at the panel the source points to (outcome K).

    python gaze/main_experiments/01_gaze_heads.py --num_pairs 500

Paper: 74.0% of cases give K. The same number of random heads from the same layers give 0.0%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("gaze_heads", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["gaze"], seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            patches = [make_patch(source, target, "query", "colon", heads=HEADS["gaze"]),
                       make_patch(source, target, "query", "colon", heads=random_heads)]
            gaze, control = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], gaze_heads=gaze == " " + letters["K"],
                        random_heads=control == " " + letters["K"], answer=gaze)
    results.report("gaze_heads", paper=74.0, label="gaze heads: pointer followed (K)")
    results.report("random_heads", paper=0.0, label="random heads: pointer followed (K)")
    results.save()


if __name__ == "__main__":
    main()
