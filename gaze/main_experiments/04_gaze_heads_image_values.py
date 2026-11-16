"""The image values of the gaze heads bring the letter (Section 6.1).

The gaze heads keep their own queries and keys, so they look at the same panel as before. But the
values they read from the image tokens are taken from the source run, which saw another image. If
the values carry the letter, the target should answer with the source's letter at the panel the
target points to (outcome C).

    python gaze/main_experiments/04_gaze_heads_image_values.py --num_pairs 500

Paper: 93.7% of cases give C.
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
    results = Results("gaze_heads_image_values", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["gaze"], seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            patches = [make_patch(source, target, "image_values", "colon", heads=HEADS["gaze"]),
                       make_patch(source, target, "image_values", "colon", heads=random_heads)]
            gaze, control = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], gaze_heads=gaze == " " + letters["C"],
                        random_heads=control == " " + letters["C"], answer=gaze)
    results.report("gaze_heads", paper=93.7, label="gaze heads read the source's image values (C)")
    results.report("random_heads", label="random heads read the source's image values (C)")
    results.save()


if __name__ == "__main__":
    main()
