"""Queries that carry the pointer (Appendix B.2, paragraph "Queries that carry the pointer").

With the 8 query heads, the query patch at the shape token moves the request (P) but sometimes the
pointer too (K): the queries of some heads already say where to look. Here we rank the heads by
how much their query moves the request MINUS how much it moves the pointer, so that heads that mix
in the pointer rank low, and apply the paper's rule for the number of heads to this ranking. Then
we patch the queries of the chosen heads at the shape token, as in main_experiments/04_query_heads.py.

By default the file uses the 128 heads that this ranking chose in an earlier, smaller run
(HEADS["query_without_pointer"]). That run ranked the heads on other pairs and tested them on 200
held-out cases, which is where the paper's numbers come from. With --find_heads the file ranks the
heads again on the search pairs of this release (two single-head searches, slow), prints the
request and pointer score of the best heads, picks the number of heads on the validation pairs,
and tests the heads it found.

    python binding/appendix_experiments/03_query_heads_without_pointer.py --num_pairs 500
    python binding/appendix_experiments/03_query_heads_without_pointer.py --find_heads

Paper: the rule needs 128 heads. They move the request (P) in 94.5% of cases and the pointer (K)
in 0.0%. Patched one at a time, L19H24 and L19H27 move the request almost alone, while L15H9,
L15H10, L17H4, L18H15 and L21H11 move the pointer about as much as the request.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like, rank_heads, search_heads, select_heads
from utils.model import load_model
from utils.patching import make_patch, run_patches

# The heads the paper names: two that move the request almost alone, five that also move the pointer.
NAMED_HEADS = [(19, 24), (19, 27), (15, 9), (15, 10), (17, 4), (18, 15), (21, 11)]


def queries(source, target, heads):
    return make_patch(source, target, "query", "shape", heads=heads)


def find_heads(model, output):
    """Rank every head by request minus pointer on the 200 search pairs, then choose K on validation."""
    scores = search_heads(model, one_answer_pairs(model, "search"), {"request": (queries, "P"), "pointer": (queries, "K")},
                          folder=Path(output) / "query_without_pointer_scores")
    difference = scores["request"] - scores["pointer"]
    print("head      request  pointer   (mean logit gain over the search cases)")
    for layer, head in NAMED_HEADS + [h for h in rank_heads(scores["request"])[:8] if h not in NAMED_HEADS]:
        print(f"L{layer}H{head:<5} {float(scores['request'][layer, head]):8.2f} {float(scores['pointer'][layer, head]):8.2f}")
    heads, _ = select_heads(model, one_answer_pairs(model, "validation"), difference, queries, "P", "request minus pointer")
    return heads


def main():
    find = "--find_heads" in sys.argv
    if find:
        sys.argv.remove("--find_heads")
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    heads = find_heads(model, args.output) if find else HEADS["query_without_pointer"]
    results = Results("query_heads_without_pointer", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(heads, seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            patched, control = run_patches(model, target, [queries(source, target, heads),
                                                           queries(source, target, random_heads)])["answers"]
            results.add(pair["index"], request=patched == " " + letters["P"], pointer=patched == " " + letters["K"],
                        random_heads=control == " " + letters["P"], answer=patched)
    print(f"{len(heads)} heads")
    results.report("request", paper=94.5, label="request moved (P)")
    results.report("pointer", paper=0.0, label="pointer moved (K)")
    results.report("random_heads", label="random heads: request moved (P)")
    results.save(heads=heads)


if __name__ == "__main__":
    main()
