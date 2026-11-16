"""The first answer token does not set the cursor (Section 5.3).

The control for the previous experiment. The target is the same: its shape at panels 1, 3, 4 and
6, the letters of panels 1 and 3 written, the letter of panel 4 next. Now the source is the image
with the letters of panels 1 and 4 swapped, so the first written letter sits in panel 4. We apply
the same patch at the first answer token: the heads read the source's image keys. If the cursor
came from this token too, the target would skip panel 4. It keeps its own answer.

    python progress/main_experiments/05_first_answer_token.py --num_pairs 500

Paper: the target keeps its own answer in 98.8% of pairs with the 16 binding heads and in 96.4%
with the 32 answer-token heads.
These pairs are drawn for this experiment (utils.dataset.progress_pairs), so --split is not used.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import name_answer, progress_pairs, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("first_answer_token", args)
    head_sets = {"binding_heads": HEADS["binding"], "answer_token_heads": HEADS["answer_token"]}
    for pair in progress_pairs(model, "cursor", args.start + args.num_pairs)[args.start:]:
        source, target = run_list(model, pair["source_first"]), run_list(model, pair["target"])
        first_answer = target.token("answers")[0]
        patches = [make_patch(source, target, "image_keys", first_answer, heads=heads) for heads in head_sets.values()]
        answers = run_patches(model, target, patches)["answers"]
        outcomes = {name: name_answer(answer, pair["letters"]) for name, answer in zip(head_sets, answers)}
        results.add(pair["index"], outcomes=outcomes, **{name: outcome == "own" for name, outcome in outcomes.items()})
    results.report("binding_heads", paper=98.8, label="binding heads: own answer kept")
    results.report("answer_token_heads", paper=96.4, label="answer-token heads: own answer kept")
    results.save()


if __name__ == "__main__":
    main()
