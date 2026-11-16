"""Progress follows the requested order (Appendix G.3.5, "Progress follows the requested order").

We add "List them from right to left." to the end of the list question, and the model lists the
letters from right to left. To keep every experiment of the paper unchanged, the panels are drawn in
reverse order (panel 1 on the right, the mirrored_row layout): panel i of the code is then the i-th
panel in the requested order, and the paper's pairs and outcomes apply as they are.

The target has its shape at r1 to r4 and has written W, its letter at r1. In the source image W sits
at r2, one matching panel further in the requested order. At the answer token, the 32 answer-token
heads read the source's image keys (as in progress/main_experiments/14). If progress follows the
requested order, the next answer moves one matching panel further, from r2 to r3.

    python generalization/main_experiments/33_right_to_left_progress.py --num_pairs 100
    python generalization/main_experiments/33_right_to_left_progress.py --layout row --wording paper   (the paper's prompt)

Pairs: the paper's list pairs, drawn and asked in the variant, kept when every no-patch check passes.
The file also prints the share of candidates that pass these checks.

Paper: with the right-to-left sentence the answer moves to r3 in 66.0% of pairs, against 73.8% with
the original prompt (the 500-pair value of progress/main_experiments/14) and 0.0% for random heads.
77.0% of candidate pairs pass every no-patch check, against 87.9% with the original prompt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pass_rate, matching_letters, run_list_variant, variant_argument, variant_list_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    variant = variant_argument(layout="mirrored_row", wording="right_to_left")
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    results = Results(f"right_to_left_progress_{variant['name']}", args)
    for pair in variant_list_pairs(model, variant, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the target's letters at r1, r2, r3, r4
        target = run_list_variant(model, pair["target"], variant, [pair["written"]])
        source = run_list_variant(model, pair["source"], variant, [pair["written"]])
        patches = [make_patch(source, target, "image_keys", "last_answer", heads=heads)
                   for heads in (HEADS["answer_token"], random_like(HEADS["answer_token"], seed=pair["index"]))]
        moved, control = run_patches(model, target, patches)["answers"]
        results.add(pair["index"], answer_token_heads=moved == " " + letters[2], random_heads=control == " " + letters[2], answer=moved)
    print(f"candidate pairs that pass every no-patch check: {list_pass_rate(variant):.1f}%   "
          f"paper: {77.0 if variant['wording'] == 'right_to_left' else 87.9}")
    right_to_left = variant["wording"] == "right_to_left"
    results.report("answer_token_heads", paper=66.0 if right_to_left else 73.8, label="answer-token heads: answer moves to r3")
    results.report("random_heads", paper=0.0, label="random heads: answer moves to r3")
    results.save(pass_rate=list_pass_rate(variant))


if __name__ == "__main__":
    main()
