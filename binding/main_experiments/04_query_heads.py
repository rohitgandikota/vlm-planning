"""Query heads carry the request (Section 4.1).

We copy the queries of the 8 query heads at the shape token from the source run into the target
run. A query says what a head looks for. If these queries carry the request ("look for the heart"),
the target should now answer the source's question on its own image: its letter inside the shape
the source asks for (outcome P).

    python binding/main_experiments/04_query_heads.py --num_pairs 500

Paper: 81.0% of cases give P, and 8.1% give K (the pointer moved instead). The same number of
random heads from the same layers give P in 0.0%.
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
    results = Results("query_heads", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["query"], seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            patches = [make_patch(source, target, "query", "shape", heads=HEADS["query"]),
                       make_patch(source, target, "query", "shape", heads=random_heads)]
            query, control = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], query_heads=query == " " + letters["P"],
                        query_heads_pointer=query == " " + letters["K"],
                        random_heads=control == " " + letters["P"], answer=query)
    results.report("query_heads", paper=81.0, label="query heads: request moved (P)")
    results.report("query_heads_pointer", paper=8.1, label="query heads: pointer moved instead (K)")
    results.report("random_heads", paper=0.0, label="random heads: request moved (P)")
    results.save()


if __name__ == "__main__":
    main()
