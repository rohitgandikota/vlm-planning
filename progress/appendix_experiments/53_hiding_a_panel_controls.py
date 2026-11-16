"""Hiding a panel from a layer on: the controls (Appendix E.5).

This extends progress/main_experiments/11_hiding_the_next_panel.py. To hide a panel from a token, we
block that token's attention to the panel's image tokens in every head, from layer l to the last
layer. Three more tests:
    a panel without the shape   W written. We hide the target's first panel that does not hold the
                                shape from every text token except the shape token, from layer l on.
                                The answer should not move on to r3.
    the first matching panel    nothing written. We hide r1 from every text token from layer l on.
                                While the choice is open the model can start from r2 instead.
    a copied pointer            nothing written. We copy the colon token's residual entering layer
                                24 from "source_first", whose matching panels are r2 and r4, and hide
                                r2 from the colon token from layer 24 on. If the copied state held
                                both of the source's panels, the target would answer from r4.

    python progress/appendix_experiments/53_hiding_a_panel_controls.py --num_pairs 500

Paper: hiding a panel without the shape moves the answer to r3 in at most 0.2% of cases at any
layer. Hiding r1 makes the model start from r2 in 43.6% of cases from the first layer, and in 18.4%
from layer 22 on. With the copied colon state and r2 hidden, the target goes back to its own r1 in
76.6% of cases and to the source's other matching panel, r4, in 3.2%. A case counts when the runs
give their own next letter with no patch (500 of 500 for each test).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import Patch, make_patch, run_patches

COPY_LAYER = 24


def hide(run, panel, tokens, first_layer):
    """Block `tokens` from the image tokens of one panel, in every head of layers first_layer to the last."""
    return Patch().block(None, tokens, run.panel_positions(panel), layers=range(first_layer, n_layers(run.model)))


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    layers = range(n_layers(model))
    results = Results("hiding_a_panel_controls", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        r, letters = pair["panels"], matching_letters(pair["target"])
        said = lambda answer, k: answer == " " + letters[k - 1]    # the answer is the target's letter at r_k
        case = {}
        # A panel without the shape, hidden from every text token except the shape token. W is written.
        target = run_list(model, pair["target"], [pair["written"]], record=False)
        if target.answer == " " + letters[1]:
            other = [p for p in range(6) if p not in r][0]
            tokens = [t for t in range(target.first_text, target.last + 1) if t != target.token("shape")]
            answers = run_patches(model, target, [hide(target, other, tokens, first) for first in layers])["answers"]
            case.update({f"other_panel_hidden_from_{first:02d}": said(a, 3) for first, a in enumerate(answers)})
        # Nothing written: hide r1, and the copied colon state with r2 hidden.
        target = run_list(model, pair["target"], [])
        source = run_list(model, pair["source_first"], [])
        if target.answer == " " + letters[0] and source.answer == " " + matching_letters(pair["source_first"])[0]:
            every = range(target.first_text, target.last + 1)
            patches = [hide(target, r[0], every, first) for first in layers]
            copied = make_patch(source, target, "residual", "colon", layer=COPY_LAYER)
            patches.append(copied.add(hide(target, r[1], [target.token("colon")], COPY_LAYER)))
            answers = run_patches(model, target, patches)["answers"]
            case.update({f"r1_hidden_from_{first:02d}": said(a, 2) for first, a in enumerate(answers[:-1])})
            case.update(copied_state_own_r1=said(answers[-1], 1), copied_state_source_r4=said(answers[-1], 4))
        results.add(pair["index"], **case)
    worst = max(layers, key=lambda first: results.rate(f"other_panel_hidden_from_{first:02d}")["percent"] or 0)
    results.report(f"other_panel_hidden_from_{worst:02d}", paper=0.2, label=f"panel without the shape, layer {worst} (the worst)")
    results.report("r1_hidden_from_00", paper=43.6, label="r1 hidden from the first layer: starts from r2")
    results.report("r1_hidden_from_22", paper=18.4, label="r1 hidden from layer 22 on: starts from r2")
    results.report("copied_state_own_r1", paper=76.6, label="copied colon state, r2 hidden: own r1")
    results.report("copied_state_source_r4", paper=3.2, label="copied colon state, r2 hidden: source's r4")
    results.save()


if __name__ == "__main__":
    main()
