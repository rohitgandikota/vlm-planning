"""On a list, the relay heads carry one panel, not the set (Appendix C.2).

The list task, with nothing written yet: the next token is the first answer. Write r1 < r2 < r3 < r4
for the target's four matching panels. The source holds the same shape at r2 and r4 only, so its
first answer is at r2 and its other panel is r4 (pair["source_first"] of utils.dataset.list_pairs).

At the colon token, the relay heads read the source's values from the shape token through the token
before the colon token. The first answer should move to r2. Do they carry only r2, or the set
{r2, r4}? We hide r2 in the target: every text token after the image is blocked from the image
tokens of panel r2, in every layer. If the set had traveled, the target would answer from r4.
The target's letter at each panel names the outcome:
    other_panel   r4, the source's other panel (the set traveled)
    own_first     r1, the target's own first panel
    next_panel    r3, the panel after the hidden one
    source_first  r2, the hidden panel is still read (hiding a panel from the text is never complete)
We repeat this with every head reading the source's values, and with random heads.

    python relay/appendix_experiments/03_relay_one_panel_not_the_set.py --num_pairs 500

Paper: the relay heads move the first answer to r2 in 78.5% of cases (random heads 0.0%). With r2
hidden: r4 8.4%, r1 31.9%, r3 14.7%, r2 30.9%. With every head: r1 48.7%, r3 22.5%, r4 4.7%.
These numbers come from an earlier, smaller run: 191 cases of an earlier list set (one direction
per pair) and an earlier relay-head list. This file runs the same test on the standard held-out
list pairs with the current relay heads, so expect values near the paper's, not the same.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, name_answer, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import all_heads, load_model, n_layers
from utils.patching import Patch, make_patch, run_patches

PAPER = {"relay_heads_moved": 78.5, "random_heads_moved": 0.0,
         "relay_heads_hidden_other_panel": 8.4, "relay_heads_hidden_own_first": 31.9,
         "relay_heads_hidden_next_panel": 14.7, "relay_heads_hidden_source_first": 30.9,
         "every_head_hidden_own_first": 48.7, "every_head_hidden_next_panel": 22.5, "every_head_hidden_other_panel": 4.7}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("relay_one_panel_not_the_set", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        source, target = run_list(model, pair["source_first"]), run_list(model, pair["target"])
        r1, r2, r3, r4 = pair["panels"]
        letters = {name: pair["target"]["letters"][panel] for name, panel in
                   (("own_first", r1), ("source_first", r2), ("next_panel", r3), ("other_panel", r4))}
        read_from = range(target.token("shape"), target.token("colon"))
        hide_r2 = Patch().block(None, range(target.first_text, target.last + 1), target.panel_positions(r2),
                                layers=range(n_layers(model)))
        head_sets = {"relay_heads": HEADS["relay"], "every_head": all_heads(model),
                     "random_heads": random_like(HEADS["relay"], seed=pair["index"])}
        patches = []                                                # per head set: the values alone, then with r2 hidden
        for heads in head_sets.values():
            values = lambda: make_patch(source, target, "values", "colon", heads=heads, positions=read_from)
            patches += [values(), values().add(hide_r2)]
        answers = run_patches(model, target, patches)["answers"]
        row = {}
        for i, name in enumerate(head_sets):
            moved, hidden = (name_answer(answer, letters) for answer in answers[2 * i:2 * i + 2])
            row[f"{name}_moved"] = moved == "source_first"          # r2: the source's first panel
            row.update({f"{name}_hidden_{outcome}": hidden == outcome for outcome in letters})
        results.add(pair["index"], **row)
    for condition, paper in PAPER.items():
        results.report(condition, paper=paper)
    results.save()


if __name__ == "__main__":
    main()
