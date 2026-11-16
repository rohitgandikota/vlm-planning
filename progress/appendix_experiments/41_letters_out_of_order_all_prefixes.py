"""Which panel select picks after letters written out of order, for all eight prefixes (Appendix E.1).

This extends progress/main_experiments/08_letters_out_of_order.py from four prefixes to the eight
rows of the appendix figure. No patch. The target has four matching panels, r1 to r4. We write some
of their letters for the model and read its next letter. Each row of the figure is one prefix, and
each cell is the share of targets whose next letter comes from r1, r2, r3, r4, the end of the turn,
or anything else.

    python progress/appendix_experiments/41_letters_out_of_order_all_prefixes.py --num_pairs 500

Paper (Figure "Select starts at the last written panel"): with r2 written, r3 in 92.8% and r1 in
2.6%; with r3 written, r4 in 64.4% and r2 in 34.4%; with r3 then r1, r2 in 95.6%; with r2 then
r1, r3 in 87.2%. The other four rows are printed in the figure only. Every held-out pair counts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model

# The letters we write, as matching panels in the order of writing (1 is r1, the leftmost matching panel).
PREFIXES = {"r1": [1], "r1_r2": [1, 2], "r2": [2], "r3": [3], "r4": [4],
            "r1_r3": [1, 3], "r2_r1": [2, 1], "r3_r1": [3, 1]}
OUTCOMES = ("r1", "r2", "r3", "r4", "end", "other")


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("letters_out_of_order_all_prefixes", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the letters at r1, r2, r3, r4
        case = {}
        for name, panels in PREFIXES.items():
            answer = run_list(model, pair["target"], [letters[p - 1] for p in panels], record=False).answer
            said = next((f"r{i + 1}" for i, letter in enumerate(letters) if answer == " " + letter),
                        "end" if answer == "<end>" else "other")
            case.update({f"{name}_says_{outcome}": said == outcome for outcome in OUTCOMES})
        results.add(pair["index"], **case)
    paper = {"r2_says_r3": 92.8, "r2_says_r1": 2.6, "r3_says_r4": 64.4, "r3_says_r2": 34.4,
             "r3_r1_says_r2": 95.6, "r2_r1_says_r3": 87.2}
    for name in PREFIXES:
        for outcome in OUTCOMES:
            condition = f"{name}_says_{outcome}"
            results.report(condition, paper=paper.get(condition), label=f"written {name.replace('_', ', ')}: next is {outcome}")
    results.save()


if __name__ == "__main__":
    main()
