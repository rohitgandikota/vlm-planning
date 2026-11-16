"""Every written letter tells the comma token what is done (Appendix E.1, table "Letters written out of order").

This extends progress/main_experiments/09_earlier_letter_blocked.py to every row of the table.
No patch from a source. We write letters out of order, "r2, r1" or "r3, r2, r1", and the model
continues with the next unwritten panel (r3 or r4). We then block the attention from the last comma
token to the answer token of one written letter (or of every written letter but the last), in
every layer and every head, and read which panel the next letter comes from: the next unwritten
panel, or a panel that was already written.

    python progress/appendix_experiments/42_comma_blocked_from_written_letters.py --num_pairs 500

Paper (next unwritten panel; a panel already written):
    r2, r1     blocked from r2: 45.4%; 52.8%       blocked from r1: 80.5%; 6.2%
    r3, r2, r1 blocked from r3: 42.5%; 57.0%       blocked from r2: 76.2%; 23.5%
               blocked from every letter but the last: 38.2%; 60.5%
Only the lists whose next letter is right with nothing blocked count: 436 of the 500 held-out
pairs after "r2, r1" and 463 after "r3, r2, r1".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import Patch, run_patches

# The written prefix (matching panels in the order of writing) and the next unwritten panel.
DESIGNS = {"r2_r1": ([2, 1], 3), "r3_r2_r1": ([3, 2, 1], 4)}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    every_layer = range(n_layers(model))
    results = Results("comma_blocked_from_written_letters", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the letters at r1, r2, r3, r4
        for name, (prefix, expected) in DESIGNS.items():
            written = [letters[p - 1] for p in prefix]
            target = run_list(model, pair["target"], written, record=False)
            if target.answer != " " + letters[expected - 1]:       # with nothing blocked the next panel must be right
                continue
            comma, answers = [target.token("last_comma")], target.token("answers")
            blocked = {f"r{p}": [answers[i]] for i, p in enumerate(prefix)}     # the answer token of each written letter
            blocked["all_but_last"] = answers[:-1]
            patches = [Patch().block(None, comma, positions, layers=every_layer) for positions in blocked.values()]
            result = run_patches(model, target, patches)["answers"]
            for cut, answer in zip(blocked, result):
                results.add(pair["index"], design=name, cut=cut, answer=answer,
                            **{f"{name}_blocked_from_{cut}_next": answer == " " + letters[expected - 1],
                               f"{name}_blocked_from_{cut}_written": answer.strip() in written})
    for name, cut, paper in (("r2_r1", "r2", (45.4, 52.8)), ("r2_r1", "r1", (80.5, 6.2)), ("r3_r2_r1", "r3", (42.5, 57.0)),
                             ("r3_r2_r1", "r2", (76.2, 23.5)), ("r3_r2_r1", "all_but_last", (38.2, 60.5))):
        results.report(f"{name}_blocked_from_{cut}_next", paper=paper[0], label=f"{name}, blocked from {cut}: next unwritten panel")
        results.report(f"{name}_blocked_from_{cut}_written", paper=paper[1], label=f"{name}, blocked from {cut}: a written panel")
    results.save()


if __name__ == "__main__":
    main()
