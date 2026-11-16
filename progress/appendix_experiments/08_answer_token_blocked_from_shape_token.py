"""The answer token needs nothing from the question (Appendix D.3, last paragraph).

The target has written W, its letter at r1, so its next answer is its letter at r2. We block one
token's attention to the shape token, in every head of every layer, and read the next letter.
    answer token blocked   if the answer token needed the question to set the cursor, the answer
                           would change
    comma token blocked    the comma token uses the shape token to find the matching panels again

    python progress/appendix_experiments/08_answer_token_blocked_from_shape_token.py --num_pairs 500

Paper: with the answer token blocked from the shape token the next answer is unchanged in 98.0% of
cases. With the comma token blocked instead, 73.2% of answers survive.
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
    results = Results("answer_token_blocked_from_shape_token", args)
    every_layer = range(n_layers(model))
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        own_next = matching_letters(pair["target"])[1]
        target = run_list(model, pair["target"], [pair["written"]])
        shape = [target.token("shape")]
        patches = [Patch().block(None, [target.token("last_answer")], shape, layers=every_layer),
                   Patch().block(None, [target.token("last_comma")], shape, layers=every_layer)]
        answer_blocked, comma_blocked = run_patches(model, target, patches)["answers"]
        results.add(pair["index"], answer_token_blocked=answer_blocked == " " + own_next,
                    comma_token_blocked=comma_blocked == " " + own_next)
    results.report("answer_token_blocked", paper=98.0, label="answer token blocked from shape token: unchanged")
    results.report("comma_token_blocked", paper=73.2, label="comma token blocked from shape token: unchanged")
    results.save()


if __name__ == "__main__":
    main()
