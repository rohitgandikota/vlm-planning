"""The top heads of the two rankings with no source (Appendix E.7, second paragraph).

The standard list pair: the target has written W, its letter at r1, and answers from r2; the source
has the shape at r2 and r4 and wrote W at r2, so its next panel is r4. For the top K heads of each
ranking (K = 4, 8, 16, 32, 64; HEADS["selection_by_zeroing"] and HEADS["selection_by_attention_shift"],
from progress/appendix_experiments/00_find_selection_rankings.py), at the last comma token:
    zeroed         the heads' outputs are set to zero in the target's own run: is the next letter unchanged?
    patched        the heads' outputs are copied from the source: does the answer move to r4?
    patched, held  the same, with the 32 gaze heads' queries held at the target's own: unchanged?
with the same number of random heads from the same layers as control.

    python progress/appendix_experiments/56_selection_rankings_without_source.py --num_pairs 500

Paper: zeroing the top 4 to 64 heads of either ranking leaves the next letter unchanged in 87.1% to
99.0% of cases. Copying their outputs moves the next panel in at most 33.5% (the top 64 by zeroing),
against at most 0.5% for random heads. With the gaze heads' queries held, every such patch leaves
the answer unchanged in 97.9% or more. These numbers come from an earlier, smaller run (200 held-out
pairs of an earlier draw of the list task, 194 valid).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import Patch, make_patch, run_patches

RANKINGS = ("selection_by_zeroing", "selection_by_attention_shift")
SIZES = (4, 8, 16, 32, 64)


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    size = model.config.text_config.head_dim
    results = Results("selection_rankings_without_source", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the target's letters at r1, r2, r3, r4
        target = run_list(model, pair["target"], [pair["written"]])
        source = run_list(model, pair["source"], [pair["written"]])
        if target.answer != " " + letters[1] or source.answer != " " + matching_letters(pair["source"])[1]:
            continue
        comma, zero = target.token("last_comma"), torch.zeros(size, device=target.device)
        hold = make_patch(target, target, "query", "last_comma", heads=HEADS["gaze"])
        names, patches = [], []
        for ranking in RANKINGS:
            for k in SIZES:
                top = HEADS[ranking][:k]
                for label, heads in (("top", top), ("random", random_like(top, seed=pair["index"] + k))):
                    zeroed = Patch()
                    for layer, head in heads:
                        zeroed.head_output(layer, head, comma, zero)
                    name = f"{ranking}_{label}_{k}"
                    names += [f"{name}_zeroed", f"{name}_patched", f"{name}_patched_held"]
                    patches += [zeroed, make_patch(source, target, "head_output", "last_comma", heads=heads),
                                make_patch(source, target, "head_output", "last_comma", heads=heads).add(hold)]
        answers = run_patches(model, target, patches)["answers"]
        own, moved = " " + letters[1], " " + letters[3]
        results.add(pair["index"], **{name + ("_moves" if name.endswith("_patched") else "_unchanged"):
                                      answer == (moved if name.endswith("_patched") else own) for name, answer in zip(names, answers)})
    for ranking in RANKINGS:
        for k in SIZES:
            for label in ("top", "random"):
                name = f"{ranking}_{label}_{k}"
                results.report(f"{name}_zeroed_unchanged", label=f"{ranking} {label} {k}: zeroed, unchanged")
                results.report(f"{name}_patched_moves", paper=33.5 if (ranking, label, k) == ("selection_by_zeroing", "top", 64) else None,
                               label=f"{ranking} {label} {k}: patched, moves to r4")
                results.report(f"{name}_patched_held_unchanged", label=f"{ranking} {label} {k}: patched, gaze held, unchanged")
    results.save()


if __name__ == "__main__":
    main()
