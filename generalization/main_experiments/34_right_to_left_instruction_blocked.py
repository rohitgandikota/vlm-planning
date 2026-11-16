"""The direction of a list comes from the instruction (Appendix G.3.5, "The direction comes from the instruction").

One row of six panels with the asked shape in four of them; r1 to r4 are the matching panels from
right to left. Each image is run with two prompts and the same written letters:
    R   the list question with "List them from right to left." added (it lists r1, r2, r3, r4)
    D   the paper's list question (it lists r4, r3, r2, r1)
In R, we block the attention from every token after the instruction to the seven tokens of the
instruction, in every head and layer. If the direction is read from the instruction, R now gives D's
answer. As a control, the same tokens are blocked from the first seven tokens of the question.
Written letters (in R's order): none, r2, r1 and r2, r2 and r3; R should answer r1, r3, r3, r4.

    python generalization/main_experiments/34_right_to_left_instruction_blocked.py

--num_pairs is the number of images. An image is used when R with nothing written answers r1 and D
answers r4; a written prefix is used when R gives its own expected answer and D's answer differs.
The images are drawn from seed 970 (the paper drew about 105 images from seeds 970 to 972).

Paper: with the instruction blocked, R gives D's answer in 86.9% to 100.0% of cases, for the first
answer and with one or two letters written. Blocking seven other question tokens gives D's answer in
at most 5.2% of cases.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import instruction_positions, make_variant, run_list_variant, sample_row_list_image
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import Patch, run_patches

PREFIXES = {"nothing_written": ([], 0), "r2_written": ([1], 2), "r1_r2_written": ([0, 1], 2), "r2_r3_written": ([1, 2], 3)}


def main():
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    right_to_left, default = make_variant("row", "right_to_left"), make_variant("row", "paper")
    every_layer = range(n_layers(model))
    results = Results("right_to_left_instruction_blocked", args)
    rng, kept = random.Random(970), 0
    while kept < args.num_pairs:
        side = sample_row_list_image(rng)
        r = [side["letters"][p] for p in side["matching"][::-1]]           # the letters at r1 .. r4
        if (run_list_variant(model, side, right_to_left, [], record=False).answer != " " + r[0]
                or run_list_variant(model, side, default, [], record=False).answer != " " + r[3]):
            continue
        row = {}
        for name, (written, expected) in PREFIXES.items():
            R = run_list_variant(model, side, right_to_left, [r[i] for i in written], record=False)
            D = run_list_variant(model, side, default, [r[i] for i in written], record=False)
            if R.answer != " " + r[expected] or D.answer == R.answer:
                continue
            instruction = instruction_positions(R)
            later = range(instruction[-1] + 1, R.last + 1)                  # every token after the instruction
            question_start = range(R.first_text + 1, R.first_text + 1 + len(instruction))
            blocked, control = run_patches(model, R, [Patch().block(None, later, instruction, layers=every_layer),
                                                      Patch().block(None, later, question_start, layers=every_layer)])["answers"]
            row.update({f"{name}_instruction_blocked": blocked == D.answer, f"{name}_control": control == D.answer})
        results.add(kept, **row)
        kept += 1
    for name in PREFIXES:
        results.report(f"{name}_instruction_blocked", paper="86.9 to 100.0", label=f"{name}: instruction blocked, gives D's answer")
        results.report(f"{name}_control", paper="at most 5.2", label=f"{name}: question tokens blocked, gives D's answer")
    results.save()


if __name__ == "__main__":
    main()
