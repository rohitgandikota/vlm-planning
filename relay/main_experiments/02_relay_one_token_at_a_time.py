"""Which token do the relay heads read the pointer from? (Section 4.2, Figure 3a)

The same patch as 01_relay_heads.py, but the 16 relay heads read the source's value at one token
only. We repeat this for every text token before the colon token and count the cases in which the
target answers K, its own letter at the panel the source points to.

    python relay/main_experiments/02_relay_one_token_at_a_time.py --num_pairs 500

Paper: 47.3% of cases give K when the token is the shape token, 9.9% for the question mark after
it, and at most 0.2% for any other token.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import make_patch, run_patches

PAPER = {"13 shape token": 47.3, "14 '?'": 9.9}          # every other token: at most 0.2


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("relay_one_token_at_a_time", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, letters in both_directions(model, pair):
            tokens = target.text_tokens()[:-1]                      # every text token before the colon token
            patches = [make_patch(source, target, "values", "colon", heads=HEADS["relay"], positions=[position])
                       for position, _ in tokens]
            answers = run_patches(model, target, patches, max_batch=32)["answers"]
            # one condition per token, named by its place after the image and its text
            names = [f"{i:02d} " + ("shape token" if position == target.token("shape") else repr(text))
                     for i, (position, text) in enumerate(tokens)]
            results.add(pair["index"], **{name: answer == " " + letters["K"] for name, answer in zip(names, answers)})
    print("relay heads read one token only: pointer moved (K)")
    for name in names:
        results.report(name, paper=PAPER.get(name))
    results.save()


if __name__ == "__main__":
    main()
