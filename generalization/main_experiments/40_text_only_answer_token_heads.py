"""With no image, the answer-token heads still set the cursor (Appendix G.3.7).

The held-out test of the 16 heads found with the panels described in words
(00_find_text_only_answer_token_heads.py, HEADS["text_only_answer_token"]). The target has written
W, its letter at r1; in the source W sits at r2. At the answer token the 16 heads read the source's
keys of the panel descriptions. If they set the cursor, the next answer skips r2 and moves to r3.
The control is the same number of random heads from the same layers.

    python generalization/main_experiments/40_text_only_answer_token_heads.py --num_pairs 100

Pairs: the paper's list pairs described in words, kept when every no-patch run is right.

Paper: the 16 heads move the next answer past the source's panel in 61% of held-out cases, against
0% for random heads.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import make_variant, matching_letters, run_list_variant, variant_list_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    text, heads = make_variant("text"), HEADS["text_only_answer_token"]
    results = Results("text_only_answer_token_heads", args)
    for pair in variant_list_pairs(model, text, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])
        target = run_list_variant(model, pair["target"], text, [pair["written"]])
        source = run_list_variant(model, pair["source"], text, [pair["written"]])
        patches = [make_patch(source, target, "image_keys", "last_answer", heads=h) for h in (heads, random_like(heads, seed=pair["index"]))]
        moved, control = run_patches(model, target, patches)["answers"]
        results.add(pair["index"], text_only_heads=moved == " " + letters[2], random_heads=control == " " + letters[2], answer=moved)
    results.report("text_only_heads", paper=61.0, label="16 heads: answer moves to r3")
    results.report("random_heads", paper=0.0, label="random heads: answer moves to r3")
    results.save()


if __name__ == "__main__":
    main()
