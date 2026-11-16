"""A new search for gaze heads on a blank canvas agrees with the paper's (Appendix G.3.2, "A new search agrees").

The search of gaze/main_experiments/00_find_gaze_heads.py, run again on one-answer pairs drawn on a
blank canvas with no panels (the shapes at random places, shared by the source and the target).
    1. On the search pairs, patch the query of one head at a time at the colon token, for every
       head, and score it by how much it raises the logit of K over the target's own letter T.
    2. On the validation pairs, patch the queries of the top K heads together, K = 1, 2, 4, ..., 128.
    3. Take the smallest K whose rate of K reaches 80%, or the best K if none does.
We then count how many of the selected heads are gaze heads of the paper.

    python generalization/main_experiments/00_find_gaze_heads_on_a_canvas.py

--num_pairs is the number of search pairs (the paper used 20), and half as many validation pairs
are used (the paper used 10). This is a head search and is slow.

Paper: the search selects 16 heads, and all 16 are among the 32 gaze heads of the paper.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import make_variant, variant_both_directions, variant_one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, choose_k, rank_heads, score_each_head
from utils.model import letter_token_ids, load_model
from utils.patching import make_patch, run_patches

SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


def main():
    args = parse_args(__doc__, split="search", num_pairs=20)
    model = load_model(args.model, args.device)
    canvas, ids = make_variant("canvas"), letter_token_ids(model)
    total, cases = 0, 0
    for pair in variant_one_answer_pairs(model, canvas, "search", args.num_pairs):
        for source, target, letters in variant_both_directions(model, pair, canvas):
            moved, own = ids[letters["K"]], ids[letters["T"]]
            total += score_each_head(model, source, target, "query", "colon", lambda logits: logits[moved] - logits[own])
            cases += 1
        print(f"pair {pair['index']} searched", flush=True)
    ranking = rank_heads(total / cases)
    results = Results("find_gaze_heads_on_a_canvas", args)
    for pair in variant_one_answer_pairs(model, canvas, "validation", max(1, args.num_pairs // 2)):
        for source, target, letters in variant_both_directions(model, pair, canvas):
            patches = [make_patch(source, target, "query", "colon", heads=ranking[:k]) for k in SIZES]
            answers = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], **{f"top_{k}": answer == " " + letters["K"] for k, answer in zip(SIZES, answers)})
    rates = {k: results.report(f"top_{k}", label=f"top {k} heads: pointer followed (K)")["percent"] / 100 for k in SIZES}
    chosen = choose_k(rates)
    shared = sorted(set(ranking[:chosen]) & set(HEADS["gaze"]))
    print(f"\nchosen K = {chosen}: {ranking[:chosen]}")
    print(f"{len(shared)} of them are among the paper's {len(HEADS['gaze'])} gaze heads   paper: 16 of 16")
    results.save(found=ranking[:chosen], ranking=ranking[:256], shared_with_paper=shared)


if __name__ == "__main__":
    main()
