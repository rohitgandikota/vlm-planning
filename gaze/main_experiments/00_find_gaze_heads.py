"""Find the gaze heads (Section 6.1).

The pointer reaches the colon token, but the letter is still in the image. A head that reads the
letter must build its query from the pointer. So we patch the query of one head at a time at the
colon token, from the source into the target, for all 1,152 heads. The search has three steps.
    1. On the search pairs, score each head by how much its patch raises the logit of K (the
       target's letter at the panel the source points to) over the target's own letter T.
    2. On the validation pairs, patch the queries of the top K heads together, for
       K = 1, 2, 4, ..., 256.
    3. Take the smallest K for which the target answers K in at least 80% of cases, or the K with
       the highest rate if none reaches 80%.

    python gaze/main_experiments/00_find_gaze_heads.py

--num_pairs is the number of search pairs (the paper uses 200), and half as many validation pairs
are used. Step 1 is slow. To run it in parts, give each part --start and --num_pairs; a part only
saves its scores, and a last run without --start reads them.

Paper: 32 gaze heads in layers 19-34, listed in utils/heads.py. No K reaches 80%: the rate is 68.5%
with 8 heads, 73.5% with 16, and 76.5% with 32, and it stays there up to 256 heads.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, choose_k, rank_heads, search_heads, sweep_k
from utils.model import load_model
from utils.patching import make_patch

SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def patch_queries(source, target, heads):
    return make_patch(source, target, "query", "colon", heads=heads)


def main():
    args = parse_args(__doc__, split="search", num_pairs=200)
    model = load_model(args.model, args.device)
    results = Results("find_gaze_heads", args)
    search_pairs = one_answer_pairs(model, "search", args.start + args.num_pairs)[args.start:]
    scores = search_heads(model, search_pairs, {"gaze": (patch_queries, "K")},
                          folder=Path(args.output) / "find_gaze_heads_scores")["gaze"]
    if args.start:                                                  # one part of step 1: nothing to select yet
        return
    ranking = rank_heads(scores)
    validation_pairs = one_answer_pairs(model, "validation", max(1, args.num_pairs // 2))
    rates = {k: rate["letter"] for k, rate in sweep_k(model, validation_pairs, ranking, patch_queries, "K", SIZES).items()}
    chosen = choose_k(rates)
    for k, rate in rates.items():
        print(f"top {k:3d} heads: the target answers K in {100 * rate:5.1f}% of validation cases" + ("   <- chosen" if k == chosen else ""))
    print(f"gaze heads found: {ranking[:chosen]}")
    print(f"same as the paper's {len(HEADS['gaze'])} gaze heads: {ranking[:chosen] == HEADS['gaze']}")
    results.save(found=ranking[:chosen], ranking=ranking[:256], k_sweep=rates, scores=scores.tolist())


if __name__ == "__main__":
    main()
