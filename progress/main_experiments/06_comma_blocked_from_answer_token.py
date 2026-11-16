"""The comma token needs the answer token (Section 5.3).

The target has written its first letter, "Answer: W,", and the comma token now chooses the second
letter. We block the attention from the comma token to the answer token, in every head of every
layer. Without the cursor that the answer token holds, the model should start again and write the
same letter W. As a control we block the comma token's attention to the colon token instead.

    python progress/main_experiments/06_comma_blocked_from_answer_token.py --num_pairs 500

Paper: the model writes the same letter again in 67.8% of pairs. With the colon token blocked
instead, 99.8% of answers are unchanged.
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
    results = Results("comma_blocked_from_answer_token", args)
    every_layer = range(n_layers(model))
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        written, own_next = pair["written"], matching_letters(pair["target"])[1]
        target = run_list(model, pair["target"], [written])
        comma = [target.token("last_comma")]
        patches = [Patch().block(None, comma, [target.token("last_answer")], layers=every_layer),
                   Patch().block(None, comma, [target.token("colon")], layers=every_layer)]
        blocked, control = run_patches(model, target, patches)["answers"]
        results.add(pair["index"], same_letter_again=blocked == " " + written,
                    colon_blocked_unchanged=control == " " + own_next, answer=blocked)
    results.report("same_letter_again", paper=67.8, label="answer token blocked: same letter again")
    results.report("colon_blocked_unchanged", paper=99.8, label="colon token blocked: answer unchanged")
    results.save()


if __name__ == "__main__":
    main()
