"""Only the gaze heads need to see the image at the colon token (Section 6.1).

We block attention to the image for every head at the tokens between the shape token and the colon
token, and for every head at the colon token except the 32 gaze heads. The shape token still sees
the image, because the pointer is resolved there. No source is used: we only ask whether the target
still answers correctly (outcome T). As a control, 32 random heads see the image in place of the
gaze heads.

The tokens before the shape token (the start of the question) are not blocked in this experiment.

    python gaze/main_experiments/03_only_gaze_heads_see_the_image.py --num_pairs 500

Paper: the target still answers T in 95.1% of cases. With random heads in their place, 4.8%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import all_heads, load_model, n_layers
from utils.patching import Patch, run_patches


def only_these_see_the_image(model, target, heads):
    """Block the image for everything after the shape token, except `heads` at the colon token."""
    shape, colon = target.token("shape"), target.token("colon")
    others = [head for head in all_heads(model) if head not in heads]
    patch = Patch()
    patch.block(None, range(shape + 1, colon), target.image_positions, layers=range(n_layers(model)))     # tokens between
    patch.block(others, [colon], target.image_positions)                                                  # the colon token
    return patch


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("only_gaze_heads_see_the_image", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["gaze"], seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            patches = [only_these_see_the_image(model, target, HEADS["gaze"]),
                       only_these_see_the_image(model, target, random_heads)]
            gaze, control = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], gaze_heads=gaze == " " + letters["T"],
                        random_heads=control == " " + letters["T"], answer=gaze)
    results.report("gaze_heads", paper=95.1, label="only the gaze heads see the image: correct (T)")
    results.report("random_heads", paper=4.8, label="only random heads see the image: correct (T)")
    results.save()


if __name__ == "__main__":
    main()
