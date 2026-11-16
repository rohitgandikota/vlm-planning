"""The first reliability check, with all 26 letters (Appendix A.7, second paragraph).

The same check as 01_task_reliability.py, but the letters come from all 26 capital letters. It
shows why the paper leaves out Q, O, I, L and J. For every wrong next token we ask two things.
Is the answer a letter that is not in the image at all (a misread letter), or the letter of
another panel (a wrong place)? And do the published Gaze Heads still look at the asked panel?

The paper's numbers come from an earlier, smaller set of 250 pairs (2,000 questions), whose layouts
were drawn by an earlier version of the image generator. This file draws 250 pairs with the
paper's generator and draws their letters from all 26, so its numbers will be close to the
paper's but not the same.

    python binding/appendix_experiments/02_task_reliability_26_letters.py --num_pairs 250

Paper: right in 98.0% of questions (case ignored). Of the errors, 61.0% are letters not in the
image (led by Q read as O, J as A, L as I, P as D), and the Gaze Heads are on the asked panel in
97.6% of them. Only 0.8% of all answers are the letter of another panel.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import one_answer_image, one_answer_question, sample_one_answer_pair_all_letters
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import run_model


def main():
    args = parse_args(__doc__, num_pairs=250)
    model = load_model(args.model, args.device)
    results = Results("task_reliability_26_letters", args)
    misreads = Counter()
    for candidate in range(args.start, args.start + args.num_pairs):
        pair = sample_one_answer_pair_all_letters(candidate)
        for side in (pair["source"], pair["target"]):
            image = one_answer_image(side)
            for panel, (shape, letter) in enumerate(zip(side["shapes"], side["letters"])):
                run = run_model(model, image, one_answer_question(shape))
                answer = run.answer.strip().upper()
                looked_at = max(range(4), key=run.panel_attention(HEADS["published_gaze"], "last").__getitem__)
                wrong = answer != letter
                not_in_image = answer not in side["letters"]
                if wrong and not_in_image:
                    misreads[f"{letter} read as {answer}"] += 1
                # the last two conditions count only the errors (None: the case does not count)
                results.add(candidate, right=not wrong, other_panel=wrong and not not_in_image,
                            error_not_in_image=not_in_image if wrong else None,
                            error_gaze_on_asked_panel=looked_at == panel if wrong else None, answer=run.answer)
    results.report("right", paper=98.0, label="next token is right (case ignored)")
    results.report("error_not_in_image", paper=61.0, label="errors: a letter not in the image")
    results.report("error_gaze_on_asked_panel", paper=97.6, label="errors: Gaze Heads on the asked panel")
    results.report("other_panel", paper=0.8, label="all answers: the letter of another panel")
    print("most common misread letters:", misreads.most_common(6))
    results.save(misreads=dict(misreads))


if __name__ == "__main__":
    main()
