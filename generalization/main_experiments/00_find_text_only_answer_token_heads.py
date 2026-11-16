"""Find the answer-token heads with no image (Appendix G.3.7).

The search of progress/main_experiments/00_find_answer_token_heads.py, on the list pairs described
in words instead of drawn (see 39_text_only_list.py). The target has written W, its letter at r1; in
the source W sits at r2. At the answer token, one head at a time reads the source's "image keys",
which here are the keys of the panel descriptions. Each head is scored by how much it raises the
logit of the target's letter at r3 over its own next letter (r2). The top K heads are then patched
together on the validation pairs, and the paper's rule picks K (80%, or the best K).

    python generalization/main_experiments/00_find_text_only_answer_token_heads.py

--num_pairs is the number of search pairs (the paper used 100); 100 validation pairs are used. This
is a head search and is slow. 40_text_only_answer_token_heads.py tests the heads on held-out pairs.

Paper: the search selects 16 heads (HEADS["text_only_answer_token"]); 7 of them are binding heads,
among them L19H24 and L19H27.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import make_variant, matching_letters, run_list_variant, variant_list_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, choose_k, rank_heads
from utils.model import letter_token_ids, load_model, n_heads, n_layers
from utils.patching import make_patch, run_patches

SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


def main():
    args = parse_args(__doc__, split="search", num_pairs=100)
    model = load_model(args.model, args.device)
    text, ids = make_variant("text"), letter_token_ids(model)
    scores = torch.zeros(n_layers(model), n_heads(model))
    search_pairs = variant_list_pairs(model, text, "search", args.num_pairs)
    for pair in search_pairs:
        letters = matching_letters(pair["target"])
        target = run_list_variant(model, pair["target"], text, [pair["written"]])
        source = run_list_variant(model, pair["source"], text, [pair["written"]])
        margin = lambda logits: logits[..., ids[letters[2]]] - logits[..., ids[letters[1]]]
        for layer in range(n_layers(model)):
            patches = [make_patch(source, target, "image_keys", "last_answer", heads=[(layer, head)]) for head in range(n_heads(model))]
            scores[layer] += (margin(run_patches(model, target, patches, max_batch=32)["logits"]) - margin(target.logits.cpu())) / len(search_pairs)
        print(f"pair {pair['index']} scored", flush=True)
    ranking = rank_heads(scores)
    results = Results("find_text_only_answer_token_heads", args)
    for pair in variant_list_pairs(model, text, "validation"):
        letters = matching_letters(pair["target"])
        target = run_list_variant(model, pair["target"], text, [pair["written"]])
        source = run_list_variant(model, pair["source"], text, [pair["written"]])
        answers = run_patches(model, target, [make_patch(source, target, "image_keys", "last_answer", heads=ranking[:k]) for k in SIZES])["answers"]
        results.add(pair["index"], **{f"top_{k}": answer == " " + letters[2] for k, answer in zip(SIZES, answers)})
    rates = {k: results.report(f"top_{k}", label=f"top {k} heads: answer moves to r3")["percent"] / 100 for k in SIZES}
    k = choose_k(rates)
    print(f"\nchosen K = {k}: {ranking[:k]}")
    print(f"{len(set(ranking[:k]) & set(HEADS['binding']))} of them are binding heads   paper: 7 of 16")
    results.save(found=ranking[:k], ranking=ranking[:256], scores=scores.tolist())


if __name__ == "__main__":
    main()
