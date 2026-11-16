"""Relay heads carry the pointer to the colon token (Section 4.2).

At the colon token, the 16 relay heads keep their own queries, keys and attention weights, but the
values they read from the shape token and the tokens after it come from the source run. If these
values carry the pointer, the target should answer with its own letter at the panel the source
points to (outcome K).

    python relay/main_experiments/01_relay_heads.py --num_pairs 500

Paper: 87.5% of cases give K. The same number of random heads from the same layers give 0.1%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("relay_heads", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["relay"], seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            # the tokens the values are read from: the shape token through the token before the colon token
            read_from = range(target.token("shape"), target.token("colon"))
            patches = [make_patch(source, target, "values", "colon", heads=HEADS["relay"], positions=read_from),
                       make_patch(source, target, "values", "colon", heads=random_heads, positions=read_from)]
            relay, control = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], relay_heads=relay == " " + letters["K"],
                        random_heads=control == " " + letters["K"], answer=relay)
    results.report("relay_heads", paper=87.5, label="relay heads: pointer moved (K)")
    results.report("random_heads", paper=0.1, label="random heads: pointer moved (K)")
    results.save()


if __name__ == "__main__":
    main()
