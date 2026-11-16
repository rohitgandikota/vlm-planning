"""Find the answer-token heads of the 32B (Appendix G.1.4, "The answer-token search on the 32B").

On the 32B, the binding heads alone move progress in few lists, so we repeat the answer-token search
of the main paper (progress/main_experiments/00_find_answer_token_heads.py) on the 32B itself. The
list pairs: the written letter W sits at r1 in the target and at r2 in the source. At the answer
token, heads read the source's image keys. If they bind W to its panel, the cursor moves from r1 to
r2 and the target's next answer moves from its letter at r2 to its letter at r3.

Step 1 scores one head at a time, for all 4,096 heads, on the search pairs, by the gain in
logit(letter at r3) - logit(letter at r2). Step 2 patches the top K heads together on the validation
pairs, for K = 1, 2, 4, ..., 4,096. A pair counts only when the 32B gives both its no-patch answers.
Step 3, the rule of the paper's 32B run: the smallest K whose rate is within 10 points of the rate
with all heads.

    python generalization/main_experiments/00_find_answer_token_heads_32b.py --model Qwen/Qwen3-VL-32B-Instruct --num_pairs 100

The paper used 100 search pairs and 50 validation pairs of the 8B's list pairs (`pairs_of_8b`).
Models: Qwen3-VL-32B on one GPU with 160 GB (a B200), or on the CPU with --device cpu (the paper's
run: 5 to 11 minutes per pair). 4,096 patched runs per search pair.

Paper: the rule selects 64 heads (1.6% of the heads), led by L45H11, 53 of them in layers 40-52.
On the validation pairs all 4,096 heads give 76.0%, the 64 heads 68.0%. These are
HEADS["answer_token_ranking_32b"][:64]. (The 80% rule of the main paper would take the best K.)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import matching_letters, pairs_of_8b, run_list
from utils.experiment import Results, parse_args
from utils.heads import heads_of_size, rank_heads
from utils.model import letter_token_ids, load_model, n_heads, n_layers
from utils.patching import make_patch, run_patches


def run_pair(model, pair):
    """Target and source with W written, or None when the model does not give both its own next letters."""
    written = [pair["written"]]
    target, source = run_list(model, pair["target"], written), run_list(model, pair["source"], written)
    letters = matching_letters(pair["target"])
    if target.answer != " " + letters[1] or source.answer != " " + matching_letters(pair["source"])[1]:
        return None
    return target, source, letters[1], letters[2]


def main():
    args = parse_args(__doc__, model="Qwen/Qwen3-VL-32B-Instruct", num_pairs=100)
    search_pairs = pairs_of_8b("list", "search", args.num_pairs, args.device)
    validation_pairs = pairs_of_8b("list", "validation", max(1, args.num_pairs // 2), args.device)
    model = load_model(args.model, args.device)
    ids = letter_token_ids(model)
    sizes = [2 ** i for i in range(20) if 2 ** i < n_layers(model) * n_heads(model)] + [n_layers(model) * n_heads(model)]

    # Step 1: one head at a time on the search pairs.
    scores, counted = torch.zeros(n_layers(model), n_heads(model)), 0
    for pair in search_pairs:
        runs = run_pair(model, pair)
        if runs is None:
            continue
        target, source, own, moved = runs
        margin = lambda logits: logits[..., ids[moved]] - logits[..., ids[own]]
        for layer in range(n_layers(model)):
            patches = [make_patch(source, target, "image_keys", "last_answer", heads=[(layer, head)]) for head in range(n_heads(model))]
            scores[layer] += margin(run_patches(model, target, patches, max_batch=32)["logits"]) - margin(target.logits.cpu())
        counted += 1
        print(f"pair {pair['index']} scored", flush=True)
    ranking = rank_heads(scores / max(counted, 1))

    # Step 2: the top K heads together on the validation pairs.
    results = Results("find_answer_token_heads_32b", args)
    for pair in validation_pairs:
        runs = run_pair(model, pair)
        if runs is None:
            continue
        target, source, own, moved = runs
        patches = [make_patch(source, target, "image_keys", "last_answer", heads=ranking[:k]) for k in sizes]
        answers = run_patches(model, target, patches, max_batch=4)["answers"]
        results.add(pair["index"], **{f"top_{k}": answer == " " + moved for k, answer in zip(sizes, answers)})
    rates = {k: results.report(f"top_{k}", label=f"top {k} heads: answer moves to r3")["percent"] for k in sizes}

    # Step 3: the smallest K within 10 points of all heads.
    k = next(size for size in sizes if rates[size] >= rates[sizes[-1]] - 10)
    print(f"\nchosen K = {k}. The heads:\n{ranking[:k]}")
    print(f"{len(set(ranking[:k]) & set(heads_of_size(args.model, 'binding')))} of the size's binding heads are among them")
    results.save(heads=ranking[:k], ranking=ranking[:256], scores=(scores / max(counted, 1)).tolist(), search_pairs_counted=counted)


if __name__ == "__main__":
    main()
