"""The first search for selection heads (Appendix E.7).

The standard list pair: the target has its shape at r1 to r4 and has written W, its letter at r1;
the source has the shape at r2 and r4 and has written W at r2, so its next panel is r4. We copy the
output of one head at a time at the last comma token from the source into the target, for every
head of layers 20-25, and score it twice: how much it raises the logit of the target's letter at r4
(the next panel moved) and at r3 (the cursor moved), over the target's own next letter, against
the unpatched target. We rank the heads by the mean score on the search pairs, patch the top K
heads together on the validation pairs for K = 1, 2, 4, ..., 128, and take the smallest K whose
rate reaches 80% (the best K if none does).

    python progress/appendix_experiments/00_find_selection_heads.py --num_pairs 100

Paper: no size reaches 80% for either outcome. The best size for the next panel, 64 heads, reaches
44%; these are HEADS["selection_output"]. That search ran on an earlier, smaller draw of the list
task (100 search pairs, 50 validation pairs); with --num_pairs 100 this file uses the first 100
search pairs and the first 50 validation pairs of the current draw.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, K_VALUES, choose_k, rank_heads
from utils.model import letter_token_ids, load_model, n_heads, n_layers
from utils.patching import make_patch, run_patches

LAYERS = range(20, 26)
SCORES = {"next_panel": 3, "cursor": 2}                            # the outcome's matching panel (index 3 is r4)


def standard_runs(model, pair):
    """The target and source runs of one pair, or None when a run does not give its own next letter."""
    target = run_list(model, pair["target"], [pair["written"]])
    source = run_list(model, pair["source"], [pair["written"]])
    ok = target.answer == " " + matching_letters(pair["target"])[1] and source.answer == " " + matching_letters(pair["source"])[1]
    return (source, target) if ok else None


def main():
    args = parse_args(__doc__, num_pairs=100)                       # the search always runs on the search split
    model = load_model(args.model, args.device)
    ids = letter_token_ids(model)
    heads = [(layer, head) for layer in LAYERS for head in range(n_heads(model))]
    total = {name: torch.full((n_layers(model), n_heads(model)), float("-inf")) for name in SCORES}
    for name in SCORES:
        total[name][list(LAYERS)] = 0.0
    cases = 0
    for pair in list_pairs(model, "search", args.start + args.num_pairs)[args.start:]:
        runs = standard_runs(model, pair)
        if runs is None:
            continue
        source, target = runs
        letters = matching_letters(pair["target"])
        logits = run_patches(model, target, [make_patch(source, target, "head_output", "last_comma", heads=[h]) for h in heads], max_batch=32)["logits"]
        own = ids[letters[1]]
        for name, k in SCORES.items():
            want = ids[letters[k]]
            gain = (logits[:, want] - logits[:, own]) - float(target.logits[want] - target.logits[own])
            total[name][list(LAYERS)] += gain.view(len(LAYERS), n_heads(model))
        cases += 1
        print(f"pair {pair['index']} searched", flush=True)
    rankings = {name: rank_heads(total[name] / max(cases, 1))[:128] for name in SCORES}
    results = Results("find_selection_heads", args)
    for pair in list_pairs(model, "validation", args.num_pairs // 2):
        runs = standard_runs(model, pair)
        if runs is None:
            continue
        source, target = runs
        letters = matching_letters(pair["target"])
        row = {}
        for name, k in SCORES.items():
            patches = [make_patch(source, target, "head_output", "last_comma", heads=rankings[name][:size]) for size in K_VALUES]
            answers = run_patches(model, target, patches)["answers"]
            row.update({f"{name}_top_{size}": answer == " " + letters[k] for size, answer in zip(K_VALUES, answers)})
        results.add(pair["index"], **row)
    for name in SCORES:
        rates = {size: results.rate(f"{name}_top_{size}")["percent"] / 100 for size in K_VALUES}
        size = choose_k(rates)
        for k in K_VALUES:
            results.report(f"{name}_top_{k}", paper=44.0 if (name, k) == ("next_panel", 64) else None,
                           label=f"{name}: top {k}" + ("   <- chosen" if k == size else ""))
        chosen = rankings[name][:size]
        print(f"{name}: {size} heads, {len(set(chosen) & set(HEADS['gaze']))} of them gaze heads, "
              f"{len(set(chosen) & set(HEADS['selection_output']))} shared with HEADS['selection_output']")
    results.save(rankings={name: ranking for name, ranking in rankings.items()})


if __name__ == "__main__":
    main()
