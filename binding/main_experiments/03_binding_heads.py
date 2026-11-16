"""Binding heads carry the pointer (Section 4.1).

We copy the outputs of the 16 binding heads at the shape token from the source run into the target
run. If these heads hold a pointer to the matching panel, the target should now answer with its own
letter at the panel the source points to (outcome K).

    python binding/main_experiments/03_binding_heads.py --num_pairs 500

Paper: 85.9% of cases give K. The same number of random heads from the same layers give 0.0%.
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
    results = Results("binding_heads", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["binding"], seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            patches = [make_patch(source, target, "head_output", "shape", heads=HEADS["binding"]),
                       make_patch(source, target, "head_output", "shape", heads=random_heads)]
            binding, control = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], binding_heads=binding == " " + letters["K"],
                        random_heads=control == " " + letters["K"], answer=binding)
    results.report("binding_heads", paper=85.9, label="binding heads: pointer moved (K)")
    results.report("random_heads", paper=0.0, label="random heads: pointer moved (K)")
    results.save()


if __name__ == "__main__":
    main()
