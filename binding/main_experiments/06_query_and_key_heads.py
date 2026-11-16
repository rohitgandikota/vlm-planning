"""Queries and image keys together move the pointer (Section 4.1).

The query heads carry the request and the key heads read the layout. Here we patch both at the
shape token, on the 10 heads that are a query head or a key head: each head gets the source's query
and scores the image tokens with the source's image keys. Request and layout now both come from
the source, so the pointer itself should move: the target answers with its own letter at the panel
the source points to (outcome K).

    python binding/main_experiments/06_query_and_key_heads.py --num_pairs 500

Paper: 95.0% of cases give K. The same number of random heads from the same layers give 0.1%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches


def queries_and_image_keys(source, target, heads):
    """Patch the queries and the image keys of the same heads at the shape token."""
    patch = make_patch(source, target, "query", "shape", heads=heads)
    return patch.add(make_patch(source, target, "image_keys", "shape", heads=heads))


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("query_and_key_heads", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["query_and_key"], seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            patches = [queries_and_image_keys(source, target, HEADS["query_and_key"]),
                       queries_and_image_keys(source, target, random_heads)]
            both, control = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], query_and_key_heads=both == " " + letters["K"],
                        random_heads=control == " " + letters["K"], answer=both)
    results.report("query_and_key_heads", paper=95.0, label="queries and image keys: pointer moved (K)")
    results.report("random_heads", paper=0.1, label="random heads: pointer moved (K)")
    results.save()


if __name__ == "__main__":
    main()
