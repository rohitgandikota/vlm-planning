"""Find the answer-token heads (Section 5.3).

After the model writes a letter, which heads at that answer token bind the letter to its panel? We
use list pairs in which the written letter W sits in another panel of the source image: at r1 in
the target and at r2 in the source. At the answer token we let heads read the source's image keys.
If they bind W to its panel, the cursor moves from r1 to r2, and the target's next answer moves
from its letter at r2 to its letter at r3.

Step 1 scores one head at a time on the 200 search pairs, by the gain in
logit(letter at r3) - logit(letter at r2). Step 2 patches the top K heads together on the 100
validation pairs, for K = 1, 2, 4, ..., 128. The paper's rule takes the smallest K that moves the
answer in 80% of cases, or the best K if none does.

    python progress/main_experiments/00_find_answer_token_heads.py        # --num_pairs sets the search pairs

Paper: no K reaches 80%, and the best is K = 32 (78%). These 32 heads are HEADS["answer_token"].
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, choose_k, rank_heads
from utils.model import letter_token_ids, load_model, n_heads, n_layers
from utils.patching import make_patch, run_patches

SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


def run_pair(model, pair):
    """Target and source runs with W written, the target's own next letter (r2), and the letter after it (r3)."""
    letters = matching_letters(pair["target"])
    written = [pair["written"]]
    return run_list(model, pair["target"], written), run_list(model, pair["source"], written), letters[1], letters[2]


def main():
    args = parse_args(__doc__, num_pairs=200)
    model = load_model(args.model, args.device)
    ids = letter_token_ids(model)

    # Step 1: one head at a time on the search pairs.
    search_pairs = list_pairs(model, "search", args.num_pairs)
    scores = torch.zeros(n_layers(model), n_heads(model))
    for pair in search_pairs:
        target, source, own, moved = run_pair(model, pair)
        margin = lambda logits: logits[..., ids[moved]] - logits[..., ids[own]]
        for layer in range(n_layers(model)):
            patches = [make_patch(source, target, "image_keys", "last_answer", heads=[(layer, head)])
                       for head in range(n_heads(model))]
            patched = run_patches(model, target, patches, max_batch=32)["logits"]
            scores[layer] += (margin(patched) - margin(target.logits.cpu())) / len(search_pairs)
        print(f"pair {pair['index']} scored", flush=True)
    ranking = rank_heads(scores)

    # Step 2: the top K heads together on the validation pairs.
    results = Results("find_answer_token_heads", args)
    for pair in list_pairs(model, "validation"):
        target, source, own, moved = run_pair(model, pair)
        patches = [make_patch(source, target, "image_keys", "last_answer", heads=ranking[:k]) for k in SIZES]
        answers = run_patches(model, target, patches)["answers"]
        results.add(pair["index"], **{f"top_{k}": answer == " " + moved for k, answer in zip(SIZES, answers)})
    rates = {k: results.report(f"top_{k}", label=f"top {k} heads: answer moves to r3")["percent"] / 100 for k in SIZES}
    k = choose_k(rates)
    print(f"\nchosen K = {k}. The heads:\n{ranking[:k]}")
    print(f"{len(set(ranking[:k]) & set(HEADS['answer_token']))} of them are among the paper's {len(HEADS['answer_token'])}")
    results.save(heads=ranking[:k], ranking=ranking[:256], scores=scores.tolist())


if __name__ == "__main__":
    main()
