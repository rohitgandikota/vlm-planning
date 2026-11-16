"""Find the heads of the value route (Appendix D.4, second paragraph).

The search behind HEADS["value_route"] in 10_value_route.py. The target has written W, its letter
at r1; the source holds W at r2. At the comma token one head reads the source's values of the
answer token. The comma token is blocked from the image except through the 32 gaze heads, in the
patched run and in the run we compare it with, so that it cannot find W in the image by itself.
    Step 1   on the search pairs, score each of the 1,152 heads alone by the gain in
             logit(letter at r3) - logit(letter at r2)
    Step 2   on the validation pairs, patch the top K heads together, K = 1, 2, 4, ..., 128; a case
             counts when the target answers with its letter at r3
    Step 3   take the smallest K that reaches 80%, or the best K if none does

    python progress/appendix_experiments/00_find_value_route_heads.py      # --num_pairs sets the search pairs

Paper: no K reaches 80% (the best, 8 heads, reaches 66%). These 8 heads are HEADS["value_route"]; two
of them are relay heads and none is a gaze head. The paper's search came from an earlier, smaller
run on 100 search pairs and 50 validation pairs drawn from another seed.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, K_VALUES, choose_k, rank_heads
from utils.model import all_heads, letter_token_ids, load_model, n_heads, n_layers
from utils.patching import Patch, make_patch, run_patches


def setup(model, pair, not_gaze):
    """The two runs, the target's letters at r2 and r3, and a function that builds the patch for a set of heads."""
    W, letters = [pair["written"]], matching_letters(pair["target"])
    target, source = run_list(model, pair["target"], W), run_list(model, pair["source"], W)
    answer, comma = [target.token("last_answer")], target.token("last_comma")

    def build(heads):
        patch = make_patch(source, target, "values", comma, heads=heads, positions=answer) if heads else Patch()
        return patch.block(not_gaze, [comma], target.image_positions)
    return target, letters[1], letters[2], build


def main():
    args = parse_args(__doc__, num_pairs=200)
    model = load_model(args.model, args.device)
    ids = letter_token_ids(model)
    gaze = set(HEADS["gaze"])
    not_gaze = [head for head in all_heads(model) if head not in gaze]

    # Step 1: one head at a time on the search pairs, compared with the blocked run with no patch.
    pairs = list_pairs(model, "search", args.num_pairs)
    scores = torch.zeros(n_layers(model), n_heads(model))
    for pair in pairs:
        target, own, moved, build = setup(model, pair, not_gaze)
        margin = lambda logits: logits[..., ids[moved]] - logits[..., ids[own]]
        blocked_only = margin(run_patches(model, target, [build([])])["logits"][0])
        for layer in range(n_layers(model)):
            patched = run_patches(model, target, [build([(layer, head)]) for head in range(n_heads(model))], max_batch=32)["logits"]
            scores[layer] += (margin(patched) - blocked_only) / len(pairs)
        print(f"pair {pair['index']} scored", flush=True)
    ranking = rank_heads(scores)

    # Steps 2 and 3: the top K heads together on the validation pairs.
    results = Results("find_value_route_heads", args)
    for pair in list_pairs(model, "validation", args.num_pairs // 2):
        target, own, moved, build = setup(model, pair, not_gaze)
        answers = run_patches(model, target, [build(ranking[:k]) for k in K_VALUES])["answers"]
        results.add(pair["index"], **{f"top_{k}": answer == " " + moved for k, answer in zip(K_VALUES, answers)})
    rates = {k: results.report(f"top_{k}", label=f"top {k} heads: answer moves to r3")["percent"] / 100 for k in K_VALUES}
    k = choose_k(rates)
    print(f"\nchosen K = {k}. The heads:\n{ranking[:k]}")
    print(f"{len(set(ranking[:k]) & set(HEADS['relay']))} relay heads, {len(set(ranking[:k]) & gaze)} gaze heads; "
          f"{len(set(ranking[:k]) & set(HEADS['value_route']))} of the paper's {len(HEADS['value_route'])}")
    results.save(heads=ranking[:k], ranking=ranking[:256], scores=scores.tolist())


if __name__ == "__main__":
    main()
