"""The answer-token searches (Appendix D.3, figure "label screens").

We score every one of the 1,152 heads alone at the answer token of the list task. The target has
written W, its letter at r1. The source holds W at r2, so a head that carries the source's binding
moves the target's next answer from its letter at r2 to its letter at r3.
    (a) the head reads the source's image keys. This is step 1 of
        progress/main_experiments/00_find_answer_token_heads.py and is not repeated here.
    (b) the head's output is the source's. Score: the gain in logit(r3) - logit(r2).
    (c) in the target's own run, the head's output is set to zero at the answer token.
        Score: the gain in logit(W) - logit(r2), a push toward writing W again.
    (d) the same zeroing at the comma token.
Each map is the mean score over the search pairs. We print the best 16 heads of each map, their
layers, and how many of them are binding heads.

    python progress/appendix_experiments/00_answer_token_searches.py      # the 200 search pairs

Paper: the heads that move progress sit in the layers of the binding heads. The best heads of the
zeroing at the answer token sit in layers 15-34. The zeroing at the comma token finds only late
heads, in layers 19-35, and none of them is a binding head.
Maps (c) and (d) came from an earlier, smaller run on 100 list pairs drawn from another seed.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, rank_heads
from utils.model import letter_token_ids, load_model, n_heads, n_layers
from utils.patching import Patch, make_patch, run_patches


def gains(model, target, build, want, own):
    """Patch one head at a time with `build(layer, head)`; the gain in logit(want) - logit(own)."""
    margin = lambda logits: logits[..., want] - logits[..., own]
    out = torch.zeros(n_layers(model), n_heads(model))
    for layer in range(n_layers(model)):
        patches = [build(layer, head) for head in range(n_heads(model))]
        out[layer] = margin(run_patches(model, target, patches, max_batch=32)["logits"]) - margin(target.logits.cpu())
    return out


def main():
    args = parse_args(__doc__, num_pairs=200, split="search")
    model = load_model(args.model, args.device)
    ids = letter_token_ids(model)
    maps = {name: torch.zeros(n_layers(model), n_heads(model)) for name in ("outputs", "zero_answer", "zero_comma")}
    pairs = list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]
    for pair in pairs:
        letters = matching_letters(pair["target"])
        W, r2, r3 = ids[pair["written"]], ids[letters[1]], ids[letters[2]]
        target = run_list(model, pair["target"], [pair["written"]])
        source = run_list(model, pair["source"], [pair["written"]])
        answer, comma = target.token("last_answer"), target.token("last_comma")
        zero = lambda token: (lambda layer, head: Patch().head_output(layer, head, token, torch.zeros_like(target.head_output(layer, head, token))))
        maps["outputs"] += gains(model, target, lambda l, h: make_patch(source, target, "head_output", answer, heads=[(l, h)]), r3, r2)
        maps["zero_answer"] += gains(model, target, zero(answer), W, r2)
        maps["zero_comma"] += gains(model, target, zero(comma), W, r2)
        print(f"pair {pair['index']} scored", flush=True)
    results = Results("answer_token_searches", args)
    binding = set(HEADS["binding"])
    for name, total in maps.items():
        best = rank_heads(total / len(pairs))[:16]
        layers = sorted({layer for layer, _ in best})
        print(f"{name:12s} best 16 heads in layers {layers[0]}-{layers[-1]}, {len(binding & set(best))} binding heads: {best}")
    print("paper: zero_answer in layers 15-34; zero_comma in layers 19-35 with no binding head")
    results.save(maps={name: (total / len(pairs)).tolist() for name, total in maps.items()})


if __name__ == "__main__":
    main()
