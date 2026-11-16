"""Gaze heads: the keys match, the values carry (Appendix F.3).

This extends gaze/main_experiments/04_gaze_heads_image_values.py. At the colon token, the 32 gaze
heads keep their own queries and read, from the image tokens,
    keys     the source's image keys, with their own values      counts when the target keeps T
    values   the source's image values, with their own keys      counts when the target answers C
    both     the source's image keys and values                  counts when the target answers C
If the query carries the pointer and the keys only match it by place, the source's keys change
nothing, because the panels sit at the same places in both images. The values bring the letter.
The control gives the same patches to 32 random heads from the same layers.

    python gaze/appendix_experiments/07_keys_and_values.py --num_pairs 500

Paper: keys 93.2% (T), values 93.7% (C), both 87.9% (C).
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
    results = Results("keys_and_values", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        sets = {"gaze": HEADS["gaze"], "random": random_like(HEADS["gaze"], seed=pair["index"])}
        for source, target, letters in both_directions(model, pair):
            row = {}
            for name, heads in sets.items():
                keys = make_patch(source, target, "image_keys", "colon", heads=heads)
                values = make_patch(source, target, "image_values", "colon", heads=heads)
                both = make_patch(source, target, "image_keys", "colon", heads=heads).add(values)
                answers = run_patches(model, target, [keys, values, both])["answers"]
                row[f"{name}_keys"] = answers[0] == " " + letters["T"]
                row[f"{name}_values"] = answers[1] == " " + letters["C"]
                row[f"{name}_both"] = answers[2] == " " + letters["C"]
            results.add(pair["index"], **row)
    results.report("gaze_keys", paper=93.2, label="gaze heads, source's image keys: own answer (T)")
    results.report("gaze_values", paper=93.7, label="gaze heads, source's image values (C)")
    results.report("gaze_both", paper=87.9, label="gaze heads, source's keys and values (C)")
    for test in ("keys", "values", "both"):
        results.report(f"random_{test}", label=f"random heads, {test}")
    results.save()


if __name__ == "__main__":
    main()
