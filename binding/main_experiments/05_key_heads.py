"""Key heads read the layout from the image keys (Section 4.1).

At the shape token, we let the 8 key heads score the image tokens with the source's image keys
instead of the target's. The heads keep their own queries and still read the target's values. An
image key says what a panel holds ("I am the panel with the star"). If the layout enters through
these keys, the target's request is now resolved on the source's layout: the target answers with
its own letter at the panel where the source image holds the target's shape (outcome R).

    python binding/main_experiments/05_key_heads.py --num_pairs 500

Paper: 83.1% of cases give R. The same number of random heads from the same layers give 0.0%.
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
    results = Results("key_heads", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["key"], seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            patches = [make_patch(source, target, "image_keys", "shape", heads=HEADS["key"]),
                       make_patch(source, target, "image_keys", "shape", heads=random_heads)]
            key, control = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], key_heads=key == " " + letters["R"],
                        random_heads=control == " " + letters["R"], answer=key)
    results.report("key_heads", paper=83.1, label="key heads: layout moved (R)")
    results.report("random_heads", paper=0.0, label="random heads: layout moved (R)")
    results.save()


if __name__ == "__main__":
    main()
