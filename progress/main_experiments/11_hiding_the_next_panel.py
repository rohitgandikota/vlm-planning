"""The next panel is chosen in layers 20-25 (Section 5.4).

The target has written one letter and should answer next from its second matching panel. We hide
that panel: every text token after the image is blocked from the image tokens of the panel, from
layer l to the last layer, for every l. No source is used. While the choice is still open, the
model can fall back on the following matching panel and answers with the third. Once the choice
is made, hiding the panel no longer moves the answer.

    python progress/main_experiments/11_hiding_the_next_panel.py --num_pairs 500

Paper: the model moves on to the following matching panel in 84.8% of cases when the panel is
hidden from the first layer, in 72.2% from layer 20 on, in 37.4% from layer 21 on, and in 2.8%
from layer 25 on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import Patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    layers = n_layers(model)
    results = Results("hiding_the_next_panel", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the letters of the four matching panels
        target = run_list(model, pair["target"], [pair["written"]], record=False)
        if target.answer != " " + letters[1]:                      # with nothing hidden the model answers with the second
            continue
        text_tokens = range(target.first_text, target.last + 1)    # every token after the image
        next_panel = target.panel_positions(pair["panels"][1])     # the image tokens of the second matching panel
        patches = [Patch().block(None, text_tokens, next_panel, layers=range(first, layers)) for first in range(layers)]
        answers = run_patches(model, target, patches)["answers"]
        results.add(pair["index"], **{f"hidden_from_layer_{first:02d}": answer == " " + letters[2]
                                      for first, answer in enumerate(answers)})
    for first, paper in ((0, 84.8), (20, 72.2), (21, 37.4), (25, 2.8)):
        results.report(f"hidden_from_layer_{first:02d}", paper=paper, label=f"hidden from layer {first} on: moves to the following panel")
    results.save()


if __name__ == "__main__":
    main()
