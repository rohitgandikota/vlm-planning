"""The second search: block one head at a time from the image (Appendix F.1).

The gaze heads were found by patching queries (gaze/main_experiments/00_find_gaze_heads.py). Other
heads might also read the image at the colon token without being moved by a query patch. So we run
a second search with the same three steps. No source is used.
    1. On the search pairs, block one head at a time from every image token at the colon token, and
       score the drop in the margin of the target's own letter T over the best of its other letters
       (K, P and R, the letters at the other asked panels of the pair).
    2. On the validation pairs, block the top K heads together, for K = 1, 2, 4, ..., 256, and count
       the cases where the target loses its own answer T.
    3. Take the smallest K for which the answer is lost in at least 80% of cases, or the K with the
       highest rate if none reaches 80%.
The query-patch search of the main text is the other curve of the same figure.

    python gaze/appendix_experiments/01_blocking_search.py

--num_pairs is the number of search pairs (the paper uses 200), and half as many validation pairs
are used. To run step 1 in parts, give each part --start and --num_pairs, then run once without --start.

Paper (Figure 9b): blocking up to 256 heads loses the target's answer in at most 2.5% of the
validation cases (1: 0.5, 2: 0.5, 4: 1.5, 8: 2.5, 16: 1.0, 32: 1.0, 64: 1.0, 128: 1.5, 256: 2.5).
The rule takes 8 heads, listed as HEADS["blocking_search"]; 02_head_sets_held_out.py tests them.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, choose_k, rank_heads
from utils.model import letter_token_ids, load_model, n_heads, n_layers
from utils.patching import Patch, run_patches

SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def own_margin(logits, ids, letters):
    """The logit of the target's own letter T minus the best logit of its letters K, P and R."""
    return logits[..., ids[letters["T"]]] - logits[..., [ids[letters[x]] for x in "KPR"]].max(dim=-1).values


def block_scores(model, pairs, folder):
    """Step 1: the mean drop of the own margin when one head is blocked, as a tensor (layers, heads)."""
    ids, total = letter_token_ids(model), torch.zeros(n_layers(model), n_heads(model))
    for pair in pairs:
        file = Path(folder) / f"pair{pair['index']:04d}.pt"
        if not file.exists():
            scores = torch.zeros_like(total)
            for source, target, letters in both_directions(model, pair):
                colon, image = target.token("colon"), target.image_positions
                for layer in range(n_layers(model)):
                    patches = [Patch().block([(layer, head)], [colon], image) for head in range(n_heads(model))]
                    logits = run_patches(model, target, patches, max_batch=32)["logits"]
                    scores[layer] += own_margin(target.logits.cpu(), ids, letters) - own_margin(logits, ids, letters)
            file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(scores, file)
        total += torch.load(file)
        print(f"pair {pair['index']} searched", flush=True)
    return total / (2 * len(pairs))


def main():
    args = parse_args(__doc__, split="search", num_pairs=200)
    model = load_model(args.model, args.device)
    results = Results("blocking_search", args)
    search_pairs = one_answer_pairs(model, "search", args.start + args.num_pairs)[args.start:]
    scores = block_scores(model, search_pairs, Path(args.output) / "blocking_search_scores")
    if args.start:                                                  # one part of step 1: nothing to select yet
        return
    ranking = rank_heads(scores)[:max(SIZES)]
    lost = {k: 0 for k in SIZES}
    for pair in one_answer_pairs(model, "validation", max(1, args.num_pairs // 2)):
        for source, target, letters in both_directions(model, pair):
            patches = [Patch().block(ranking[:k], [target.token("colon")], target.image_positions) for k in SIZES]
            answers = run_patches(model, target, patches)["answers"]
            for k, answer in zip(SIZES, answers):
                lost[k] += answer != " " + letters["T"]
            results.add(pair["index"], **{f"top{k}_own_answer_lost": a != " " + letters["T"] for k, a in zip(SIZES, answers)})
    rates = {k: lost[k] / len(results.rows) for k in SIZES}
    chosen = choose_k(rates)
    for k in SIZES:
        results.report(f"top{k}_own_answer_lost", label=f"top {k:3d} heads blocked: own answer lost" + ("   <- chosen" if k == chosen else ""))
    print("paper: at most 2.5% for every K; 8 heads chosen")
    print(f"same as the paper's blocking-search heads: {ranking[:chosen] == HEADS['blocking_search']}")
    results.save(found=ranking[:chosen], ranking=ranking, k_sweep=rates, scores=scores.tolist())


if __name__ == "__main__":
    main()
