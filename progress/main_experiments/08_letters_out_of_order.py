"""Select takes the nearest unwritten panel after the cursor (Section 5.4).

No patch. The target has four matching panels. We write some of their letters for the model in an
order it would not use itself, and read its next letter. If select starts at the cursor (the panel
of the last written letter) and prefers the right, the model continues after the last written
panel, and goes back only when nothing is left to the right. Panels that were written earlier
should also count as done.

    python progress/main_experiments/08_letters_out_of_order.py --num_pairs 500

Paper: with only the second matching letter written, the model answers with the third in 92.8% of
lists and goes back to the first in 2.6%. With only the third written, it answers with the fourth
in 64.4% and goes back to the second in 34.4%. With the third and then the first written, it
answers with the second in 95.6%. With the second and then the first written, it answers with the
third in 87.2%. Every held-out pair counts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model

# The letters we write, as matching panels in the order of writing (1 is the leftmost matching panel).
WRITTEN = {"second": [2], "third": [3], "third_then_first": [3, 1], "second_then_first": [2, 1]}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("letters_out_of_order", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the letters of the four matching panels
        said = {}                                                   # which matching panel the next letter comes from
        for name, panels in WRITTEN.items():
            answer = run_list(model, pair["target"], [letters[p - 1] for p in panels], record=False).answer
            said[name] = next((p + 1 for p, letter in enumerate(letters) if answer == " " + letter), 0)
        results.add(pair["index"],
                    second_says_third=said["second"] == 3, second_back_to_first=said["second"] == 1,
                    third_says_fourth=said["third"] == 4, third_back_to_second=said["third"] == 2,
                    third_then_first_says_second=said["third_then_first"] == 2,
                    second_then_first_says_third=said["second_then_first"] == 3,
                    **{"next_after_" + name: panel for name, panel in said.items()})      # saved with the row, not counted
    results.report("second_says_third", paper=92.8, label="second written: answers with the third")
    results.report("second_back_to_first", paper=2.6, label="second written: goes back to the first")
    results.report("third_says_fourth", paper=64.4, label="third written: answers with the fourth")
    results.report("third_back_to_second", paper=34.4, label="third written: goes back to the second")
    results.report("third_then_first_says_second", paper=95.6, label="third, then first: answers with the second")
    results.report("second_then_first_says_third", paper=87.2, label="second, then first: answers with the third")
    results.save()


if __name__ == "__main__":
    main()
