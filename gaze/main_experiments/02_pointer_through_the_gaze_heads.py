"""The pointer acts through the gaze heads (Section 6.1).

We repeat the binding-head patch of Section 4.1: the outputs of the 16 binding heads at the shape
token are copied from the source, which moves the pointer to the source's panel. This time we also
keep the queries of the 32 gaze heads at the colon token at the values of the unpatched target. If
the pointer acts through these queries, the patch should lose its effect and the target should keep
its own answer (outcome T). Keeping the queries of random heads should let the pointer through (K).

    python gaze/main_experiments/02_pointer_through_the_gaze_heads.py --num_pairs 500

Paper: with the gaze-head queries kept, 98.4% of cases give T. With the queries of the same number
of random heads from the same layers kept, the patched pointer gets through in 86.5% (K).
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
    results = Results("pointer_through_the_gaze_heads", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["gaze"], seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            patches = []
            for kept in (HEADS["gaze"], random_heads):
                patch = make_patch(source, target, "head_output", "shape", heads=HEADS["binding"])   # move the pointer
                patch.add(make_patch(target, target, "query", "colon", heads=kept))                  # keep these queries
                patches.append(patch)
            gaze, control = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], gaze_queries_kept=gaze == " " + letters["T"],
                        random_queries_kept=control == " " + letters["K"], answer=gaze, answer_random=control)
    results.report("gaze_queries_kept", paper=98.4, label="gaze-head queries kept: own answer (T)")
    results.report("random_queries_kept", paper=86.5, label="random-head queries kept: pointer through (K)")
    results.save()


if __name__ == "__main__":
    main()
