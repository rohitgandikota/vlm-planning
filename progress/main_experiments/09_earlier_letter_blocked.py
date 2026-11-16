"""A panel that was written earlier counts as done through its answer token (Section 5.4).

We write the second matching letter and then the first ("Answer: r2, r1,"). The model continues
with the third. We then block the attention from the last comma token to the answer token of the
earlier letter, r2, in every layer and every head. No source is used. If the comma token learns
from that answer token that its panel is done, the model should now write that letter again.

    python progress/main_experiments/09_earlier_letter_blocked.py --num_pairs 500

Paper: the model writes the earlier letter again in 52.8% of cases. Only the lists whose next
letter is right with nothing blocked are counted (436 of the 500 held-out pairs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import Patch, run_patch


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("earlier_letter_blocked", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the letters of the four matching panels
        target = run_list(model, pair["target"], [letters[1], letters[0]], record=False)
        if target.answer != " " + letters[2]:                      # with no block the model must answer with the third
            continue
        earlier_letter = target.token("answers")[0]                # the answer token of r2, written first
        patch = Patch().block(None, [target.token("last_comma")], [earlier_letter], layers=range(n_layers(model)))
        answer = run_patch(model, target, patch)
        results.add(pair["index"], writes_it_again=answer == " " + letters[1], answer=answer)
    results.report("writes_it_again", paper=52.8, label="earlier letter blocked: written again")
    results.save()


if __name__ == "__main__":
    main()
