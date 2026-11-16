"""The answer-token heads of the 32B (Appendix G.1.4, Table "sizes-answer-token").

The list pairs of the answer-token search (00_find_answer_token_heads_32b.py): with one letter W
written, W sits at r1 in the target and at r2 in the source. At the last answer token a set of heads
reads the source's image keys. If they bind W to the source's panel, the next answer moves to the
target's letter at r3, the letter after the source's panel. The head sets:
    the 64 answer-token heads of the 32B, and 64 random heads from the same layers that are not
    among the best 128 of the ranking (seed = pair index)
    the best 256 heads of the ranking, and all 4,096 heads
    the 16 binding heads of the 32B, and the 64 heads without the binding heads
A pair counts only when the 32B gives both its no-patch answers.

    python generalization/main_experiments/11_answer_token_heads_32b.py --model Qwen/Qwen3-VL-32B-Instruct --num_pairs 50

Models: Qwen3-VL-32B on one GPU with 160 GB (a B200), or on the CPU with --device cpu (the paper's
run). 6 patched runs per pair. The paper used 50 held-out list pairs.

Paper: 64 heads 68.0%; 64 random heads 0.0%; best 256 heads 68.0%; all heads 68.0%; 16 binding heads
22.0%; the 64 without the binding heads 0.0%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import matching_letters, pairs_of_8b, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, heads_of_size, random_heads_like
from utils.model import all_heads, load_model
from utils.patching import make_patch, run_patches

PAPER = {"top_64": 68.0, "random_64": 0.0, "top_256": 68.0, "all_heads": 68.0, "binding_heads": 22.0, "top_64_without_binding": 0.0}


def main():
    args = parse_args(__doc__, model="Qwen/Qwen3-VL-32B-Instruct", num_pairs=50)
    pairs = pairs_of_8b("list", args.split, args.start + args.num_pairs, args.device)[args.start:]
    model = load_model(args.model, args.device)
    ranking, binding = HEADS["answer_token_ranking_32b"], heads_of_size(args.model, "binding")
    top_64 = heads_of_size(args.model, "answer_token")
    results = Results("answer_token_heads_32b", args)
    for pair in pairs:
        written = [pair["written"]]
        target, source = run_list(model, pair["target"], written), run_list(model, pair["source"], written)
        letters = matching_letters(pair["target"])
        if target.answer != " " + letters[1] or source.answer != " " + matching_letters(pair["source"])[1]:
            continue                                                          # the 32B gets the pair wrong
        head_sets = {"top_64": top_64, "random_64": random_heads_like(model, top_64, seed=pair["index"], exclude=ranking[:128]),
                     "top_256": ranking[:256], "all_heads": all_heads(model), "binding_heads": binding,
                     "top_64_without_binding": [head for head in top_64 if head not in binding]}
        patches = [make_patch(source, target, "image_keys", "last_answer", heads=heads) for heads in head_sets.values()]
        answers = run_patches(model, target, patches, max_batch=2)["answers"]
        results.add(pair["index"], **{name: answer == " " + letters[2] for name, answer in zip(head_sets, answers)})
    for name, paper in PAPER.items():
        results.report(name, paper=paper, label=f"{name.replace('_', ' ')}: next answer follows the source")
    results.save()


if __name__ == "__main__":
    main()
