"""Find the stop heads (Section 5.4).

At the last written letter the model chooses between a comma (the list goes on) and the end of its
turn (the list stops). We search for the heads whose outputs at that answer token carry the
choice. Each list pair gives two cases: a finished list patched from a continuing one, and a
continuing list patched from a finished one. The search covers all 1,152 heads and has three steps.
    1. On the search pairs, copy the output of one head at a time from the source and score the
       head by how much the patch raises the logit of the source's choice over the target's own.
    2. On the validation pairs, patch the top K heads together for K = 1, 2, 4, ..., 128.
    3. Take the smallest K for which the target makes the source's choice in at least 80% of
       cases (the best K if none does).

    python progress/main_experiments/00_find_stop_heads.py

--num_pairs is the number of search pairs (the paper uses 200), and half as many validation pairs
are used. Step 1 runs 2,304 patches per pair, so it is slow.

Paper: 64 stop heads (98% of validation cases; 32 heads reach 58%), listed in utils/heads.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, stop_directions
from utils.experiment import Results, parse_args
from utils.heads import HEADS, K_VALUES, choose_k, rank_heads, score_each_head
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__, split="search", num_pairs=200)
    model = load_model(args.model, args.device)
    results = Results("find_stop_heads", args)
    scores, cases = 0, 0
    for pair in list_pairs(model, "search", args.start + args.num_pairs)[args.start:]:
        for _, source, target in stop_directions(model, pair):
            # How far the next-token logits lean toward the source's choice.
            lean = lambda logits: logits[source.answer_id] - logits[target.answer_id]
            scores = scores + score_each_head(model, source, target, "head_output", "last_answer", lean)
            cases += 1
        print(f"pair {pair['index']} searched", flush=True)
    ranking = rank_heads(scores / cases)

    hits, cases = {k: 0 for k in K_VALUES}, 0
    for pair in list_pairs(model, "validation", max(1, args.num_pairs // 2)):
        for _, source, target in stop_directions(model, pair):
            patches = [make_patch(source, target, "head_output", "last_answer", heads=ranking[:k]) for k in K_VALUES]
            for k, answer in zip(K_VALUES, run_patches(model, target, patches)["answers"]):
                hits[k] += answer == source.answer
            cases += 1
    rates = {k: hits[k] / cases for k in K_VALUES}
    chosen = choose_k(rates)
    for k in K_VALUES:
        print(f"  K = {k:3d}   {100 * rates[k]:5.1f}%" + ("   <- chosen" if k == chosen else ""))
    print(f"stop heads found: {ranking[:chosen]}")
    print(f"same as the paper's {len(HEADS['stop'])} stop heads: {ranking[:chosen] == HEADS['stop']}")
    # The random control of 15_stopping.py draws its heads outside the best 2K heads of this ranking.
    results.save(heads=ranking[:chosen], ranking=ranking[:2 * chosen], k_sweep=rates)


if __name__ == "__main__":
    main()
