"""The cursor is set at the last answer token (Section 5.3).

Does an answer token set the cursor? The target has its shape at panels 1, 3, 4 and 6 and has
written the letters of panels 1 and 3, so its next answer is the letter of panel 4. The source is
the same image with the letters of panels 3 and 4 swapped: the last written letter now sits one
panel to the right. At the last answer token we let a set of heads read the source's image keys.
Their queries, the values, and everything else stay the target's. If these heads bind the written
letter to its panel, the cursor moves to panel 4 and the target answers with the letter of panel 6.

    python progress/main_experiments/04_cursor_at_the_last_answer_token.py --num_pairs 500

Paper: the letter of panel 6 in 84.2% of pairs with the 16 binding heads, in 0.0% with the same
number of random heads from the same layers (the largest of five draws), and in 88.0% with the 32
answer-token heads. The target repeats a written letter in at most 2.4% of pairs.
These pairs are drawn for this experiment (utils.dataset.progress_pairs), so --split is not used.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import name_answer, progress_pairs, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("cursor_at_the_last_answer_token", args)
    head_sets = {"binding_heads": HEADS["binding"], "answer_token_heads": HEADS["answer_token"]}
    for draw in range(5):                               # the same five random sets for every pair
        head_sets[f"random_heads_{draw}"] = random_like(HEADS["binding"], seed=draw)
    for pair in progress_pairs(model, "cursor", args.start + args.num_pairs)[args.start:]:
        source, target = run_list(model, pair["source"]), run_list(model, pair["target"])
        patches = [make_patch(source, target, "image_keys", "last_answer", heads=heads) for heads in head_sets.values()]
        answers = run_patches(model, target, patches)["answers"]
        outcomes = {name: name_answer(answer, pair["letters"]) for name, answer in zip(head_sets, answers)}
        results.add(pair["index"], outcomes=outcomes,
                    **{name: outcome == "after_panel_4" for name, outcome in outcomes.items()},
                    binding_heads_repeat=outcomes["binding_heads"].startswith("repeat"),
                    answer_token_heads_repeat=outcomes["answer_token_heads"].startswith("repeat"))
    results.report("binding_heads", paper=84.2, label="binding heads: letter of panel 6")
    results.report("answer_token_heads", paper=88.0, label="answer-token heads: letter of panel 6")
    for draw in range(5):
        results.report(f"random_heads_{draw}", paper=0.0, label=f"random heads, draw {draw}: letter of panel 6")
    results.report("binding_heads_repeat", paper="at most 2.4", label="binding heads: repeats a written letter")
    results.report("answer_token_heads_repeat", paper="at most 2.4", label="answer-token heads: repeats a written letter")
    results.save()


if __name__ == "__main__":
    main()
