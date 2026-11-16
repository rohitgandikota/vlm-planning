"""The list-task blocks at every model size (Appendix G.1.4, the paragraph after Table "sizes-progress").

The target has written its first letter, "Answer: W,", and the comma token chooses the second letter.
In the target's own run, in every head of every layer, we block:
    the comma token from the answer token        the model should write W again (a repeat)
    the comma token from the colon token         the control: the answer should not change
    the answer token from the image              the comma token can still read the image
    the comma token from the image               the answer token still holds its panel
    both from the image                          both routes cut: the answer should be lost
The last four count the cases that keep the target's own second letter. A pair counts only when this
size gives the right second letter with no patch in both runs of the pair (target and source), as in
the paper's runs.

    python generalization/main_experiments/10_sizes_list_blocks.py --model Qwen/Qwen3-VL-4B-Instruct --num_pairs 500

Models: any Qwen3-VL size. 2B 16 GB, 4B 24 GB, 8B 48 GB, 32B 160 GB (one B200) or --device cpu.
5 patched runs per pair.

Paper (an earlier run, 200 held-out pairs on the 2B and 4B and 50 on the 32B): repeats 6.3 (2B),
41.6 (4B), 100.0 (32B), 67.8 (8B); one of the two blocked from the image keeps 81% or more (32B: 90.0
for the answer token, 72.0 for the comma token); both blocked keeps 12.5 (2B), 4.9 (4B), 14.0 (32B).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import matching_letters, pairs_of_8b, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import Patch, run_patches

PAPER = {"same_letter_again": {"2B": 6.3, "4B": 41.6, "8B": 67.8, "32B": 100.0},
         "answer_token_blind": {"32B": 90.0}, "comma_token_blind": {"32B": 72.0},
         "both_blind": {"2B": 12.5, "4B": 4.9, "32B": 14.0}}


def main():
    args = parse_args(__doc__)
    pairs = pairs_of_8b("list", args.split, args.start + args.num_pairs, args.device)[args.start:]
    model = load_model(args.model, args.device)
    every_layer = range(n_layers(model))
    results = Results(f"sizes_list_blocks_{args.model.split('/')[-1]}", args)
    for pair in pairs:
        written = pair["written"]
        target, source = run_list(model, pair["target"], [written]), run_list(model, pair["source"], [written])
        own_next = " " + matching_letters(pair["target"])[1]
        if target.answer != own_next or source.answer != " " + matching_letters(pair["source"])[1]:
            continue                                                          # this size gets the pair wrong
        comma, answer, image = [target.token("last_comma")], [target.token("last_answer")], target.image_positions
        patches = [Patch().block(None, comma, answer, layers=every_layer),
                   Patch().block(None, comma, [target.token("colon")], layers=every_layer),
                   Patch().block(None, answer, image, layers=every_layer),
                   Patch().block(None, comma, image, layers=every_layer),
                   Patch().block(None, answer + comma, image, layers=every_layer)]
        repeat, control, answer_blind, comma_blind, both = run_patches(model, target, patches)["answers"]
        results.add(pair["index"], same_letter_again=repeat == " " + written, colon_blocked_unchanged=control == own_next,
                    answer_token_blind=answer_blind == own_next, comma_token_blind=comma_blind == own_next, both_blind=both == own_next)
    size = args.model.split("-")[-2]
    results.report("same_letter_again", paper=PAPER["same_letter_again"].get(size), label="comma blocked from the answer token: repeat")
    results.report("colon_blocked_unchanged", label="comma blocked from the colon token: unchanged")
    results.report("answer_token_blind", paper=PAPER["answer_token_blind"].get(size), label="answer token blocked from the image: kept")
    results.report("comma_token_blind", paper=PAPER["comma_token_blind"].get(size), label="comma token blocked from the image: kept")
    results.report("both_blind", paper=PAPER["both_blind"].get(size), label="both blocked from the image: kept")
    results.save()


if __name__ == "__main__":
    main()
