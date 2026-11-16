"""The cursor marks a panel, not a letter (Section 5.3).

If the cursor were the written letter and not its panel, the model would skip a second copy of
that letter in the image. There is no patch here. We put the first written letter W into the third
matching panel as well, and write two letters for the model: "Answer: W, X,". The next matching
panel now holds W again. Does the model write W a second time? As a control we run the unchanged
image, whose third matching panel holds a new letter.

    python progress/main_experiments/07_same_letter_in_next_panel.py --num_pairs 500

Paper: the model writes W again in 95.8% of lists. On the unchanged image it writes the new letter
in 96.6%. Only the held-out pairs have the image with the repeated letter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("same_letter_in_next_panel", args)
    for pair in list_pairs(model, "heldout", args.start + args.num_pairs)[args.start:]:
        first, second, third, _ = matching_letters(pair["target"])           # first is the letter W
        repeated = run_list(model, pair["target_repeat"], [first, second], record=False).answer
        unchanged = run_list(model, pair["target"], [first, second], record=False).answer
        results.add(pair["index"], writes_letter_again=repeated == " " + first,
                    new_letter_written=unchanged == " " + third, answer=repeated)
    results.report("writes_letter_again", paper=95.8, label="next panel holds W: W written again")
    results.report("new_letter_written", paper=96.6, label="next panel holds a new letter: written")
    results.save()


if __name__ == "__main__":
    main()
