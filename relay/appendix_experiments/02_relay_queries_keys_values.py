"""Queries and keys route, values carry (Appendix C.2).

To see which part of the relay heads carries the pointer, we patch their queries, keys and values
at the colon token one at a time, from the source run.
    values   the 16 relay heads read the source's values from the shape token through the token
             before the colon token (relay/main_experiments/01_relay_heads.py). Effect: K.
    queries  their queries at the colon token come from the source. If they only route, the
             target keeps its answer T.
    keys     they score the same text tokens with the source's keys. Expected: T.
    both     queries and keys together. Expected: T.
K is the target's own letter at the panel the source points to, T is the target's own answer.

    python relay/appendix_experiments/02_relay_queries_keys_values.py --num_pairs 500

Paper: values move the pointer in 87.5% of cases. The answer stays T in 86.9% with the source's
queries, 99.2% with the source's keys, and 98.4% with both.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("relay_queries_keys_values", args)
    relay = HEADS["relay"]
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, letters in both_directions(model, pair):
            # the text tokens the relay heads read: the shape token through the token before the colon token
            text = range(target.token("shape"), target.token("colon"))
            queries = make_patch(source, target, "query", "colon", heads=relay)
            keys = make_patch(source, target, "keys", "colon", heads=relay, positions=text)
            values = make_patch(source, target, "values", "colon", heads=relay, positions=text)
            both = make_patch(source, target, "query", "colon", heads=relay).add(keys)    # the keys are scored with the new queries
            answers = run_patches(model, target, [values, queries, keys, both])["answers"]
            moved, *kept = answers
            results.add(pair["index"], values=moved == " " + letters["K"],
                        **{name: answer == " " + letters["T"] for name, answer in zip(("queries", "keys", "both"), kept)})
    results.report("values", paper=87.5, label="values: pointer moved (K)")
    results.report("queries", paper=86.9, label="queries: own answer kept (T)")
    results.report("keys", paper=99.2, label="keys: own answer kept (T)")
    results.report("both", paper=98.4, label="queries and keys: own answer kept (T)")
    results.save()


if __name__ == "__main__":
    main()
