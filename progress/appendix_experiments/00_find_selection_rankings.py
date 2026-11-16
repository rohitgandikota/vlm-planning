"""The two searches for selection heads that need no source (Appendix E.7, second paragraph).

The target has written W, its letter at r1, and answers next from r2. For every head of layers
16-25 at the last comma token we rank the heads in two ways:
    zeroing          the head's output is set to zero. Score: how much this lowers the margin of the
                     target's letter at r2 over the best of its letters at r3 and r4.
    attention shift  no patch. Score: the share of the head's attention on r2 minus its mean share on
                     r3 and r4. As in progress/main_experiments/12, the target is drawn again with
                     every shape at its panel center, and a share is of the head's whole attention.
Each score is averaged over the pairs. progress/appendix_experiments/56_selection_rankings_without_source.py
tests the top heads of both rankings.

    python progress/appendix_experiments/00_find_selection_rankings.py --num_pairs 100

Paper: the top heads of the two rankings overlap (L21H11, L20H29, L20H30, L21H18, L23H26 in both);
the best 64 of each are HEADS["selection_by_zeroing"] and HEADS["selection_by_attention_shift"].
These rankings come from an earlier, smaller run: zeroing on its first 100 pairs, attention shift on
its 200 held-out pairs. This file ranks both on the first --num_pairs pairs of --split.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import rank_heads
from utils.model import letter_token_ids, load_model, n_heads, n_layers
from utils.patching import Patch, attention_share, run_patches

LAYERS = range(16, 26)


def main():
    args = parse_args(__doc__, num_pairs=100, split="search")
    model = load_model(args.model, args.device)
    ids, size = letter_token_ids(model), model.config.text_config.head_dim
    heads = [(layer, head) for layer in LAYERS for head in range(n_heads(model))]
    scores = {name: torch.full((n_layers(model), n_heads(model)), float("-inf")) for name in ("zeroing", "attention_shift")}
    for name in scores:
        scores[name][list(LAYERS)] = 0.0
    pairs = list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]
    for pair in pairs:
        r, letters = pair["panels"], matching_letters(pair["target"])
        target = run_list(model, pair["target"], [pair["written"]], record=False)
        comma = target.token("last_comma")
        zero = torch.zeros(size, device=target.device)
        logits = run_patches(model, target, [Patch()] + [Patch().head_output(l, h, comma, zero) for l, h in heads], max_batch=32)["logits"]
        margin = logits[:, ids[letters[1]]] - torch.maximum(logits[:, ids[letters[2]]], logits[:, ids[letters[3]]])
        scores["zeroing"][list(LAYERS)] += (margin[0] - margin[1:]).view(len(LAYERS), n_heads(model))
        centered = run_list(model, {**pair["target"], "centers": None}, [pair["written"]])
        share = attention_share(centered, heads, "last_comma")
        scores["attention_shift"][list(LAYERS)] += (share[:, r[1]] - share[:, r[2:]].mean(1)).view(len(LAYERS), n_heads(model))
        print(f"pair {pair['index']} ranked", flush=True)
    rankings = {name: rank_heads(total / len(pairs))[:64] for name, total in scores.items()}
    for name, ranking in rankings.items():
        print(f"{name}: " + ", ".join(f"L{l}H{h}" for l, h in ranking[:16]))
    both = [f"L{l}H{h}" for l, h in rankings["zeroing"][:16] if (l, h) in rankings["attention_shift"][:16]]
    print("in the top 16 of both:", ", ".join(both))
    results = Results("find_selection_rankings", args)
    results.save(rankings=rankings)


if __name__ == "__main__":
    main()
