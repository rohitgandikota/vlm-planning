"""On a blank canvas, a place cannot be hidden (Appendix G.3.2, "What the panels give us").

progress/main_experiments/11 hides the next panel from every text token and watches the model fall
to the following one. Here we do the same on list pairs drawn on a blank canvas. The target has
written W, its letter at r1, and should answer next from r2. With nothing patched, we hide r2 from
every text token after the image, in every layer and head, in two ways:
    box        the image tokens inside the box around the shape at r2 (24 pixels of margin)
    nearest    every image token closer to the shape at r2 than to any other shape
and ask whether the model still names the letter at r2.

    python generalization/main_experiments/30_canvas_hiding_a_place.py

Pairs: the list pairs of the paper drawn on a canvas, with places numbered in reading order (see
27_list_task_on_other_images.py). The paper's numbers come from an earlier, smaller run (19 held-out
pairs). The default here is 100 pairs.

Paper: with the box hidden the model still names the letter of r2 in 100.0% of cases; with the
nearest image tokens hidden, in 84.2%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import make_variant, matching_letters, run_list_variant, variant_list_pairs
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import Patch, nearest_place_positions, run_patches


def main():
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    canvas = make_variant("canvas")
    results = Results("canvas_hiding_a_place", args)
    for pair in variant_list_pairs(model, canvas, args.split, args.start + args.num_pairs)[args.start:]:
        target = run_list_variant(model, pair["target"], canvas, [pair["written"]], record=False)
        r2, own = pair["panels"][1], " " + matching_letters(pair["target"])[1]
        text_tokens, every_layer = range(target.first_text, target.last + 1), range(n_layers(model))
        box, nearest = target.panel_positions(r2), nearest_place_positions(target, pair["target"]["canvas_centers"], r2)
        answers = run_patches(model, target, [Patch().block(None, text_tokens, box, layers=every_layer),
                                              Patch().block(None, text_tokens, nearest, layers=every_layer)])["answers"]
        results.add(pair["index"], box_hidden_still_read=answers[0] == own, nearest_hidden_still_read=answers[1] == own,
                    answers=answers)
    results.report("box_hidden_still_read", paper=100.0, label="box around r2 hidden: still names it")
    results.report("nearest_hidden_still_read", paper=84.2, label="nearest image tokens hidden: still names it")
    results.save()


if __name__ == "__main__":
    main()
