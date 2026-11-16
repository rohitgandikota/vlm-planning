"""A small set of heads reads the order instruction (Appendix G.3.5, Table 16).

The held-out test of the heads found by 00_find_instruction_heads.py (HEADS["instruction"], 16 heads
in layers 18-23). R is the run with "List them from right to left." and the letter of r2 written
(it answers r3); D is the run with the paper's prompt on the same image and letter. In R we block
the attention from every token after the instruction to the instruction, in a set of heads, and ask
whether R now gives D's answer (the next answer of the left-to-right order):
    none; every head; the 16 heads; every head except the 16; 16 random heads from the same layers
    (three draws); and the 16 heads blocked from the first seven question tokens instead.

    python generalization/main_experiments/35_instruction_heads_blocked.py

--num_pairs is the number of held-out images (the paper used 104), drawn from seed 988 as in the
search (an image is used when R answers r3 and D answers something else).

Paper: none 0.0%; every head 100.0%; the 16 heads 98.4%; every head except the 16, 0.0% (R stays right
to left in 98.4%); random heads at most 1.6%; the 16 heads blocked from other question tokens 0.0%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import instruction_positions, right_to_left_items
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import all_heads, load_model, n_layers
from utils.patching import Patch, run_patches


def main():
    args = parse_args(__doc__, num_pairs=104)
    model = load_model(args.model, args.device)
    heads = HEADS["instruction"]
    others = [head for head in all_heads(model) if head not in heads]
    results = Results("instruction_heads_blocked", args)
    for index, (R, D) in enumerate(right_to_left_items(model, 988, args.num_pairs)):
        instruction = instruction_positions(R)
        later = range(instruction[-1] + 1, R.last + 1)                       # every token after the instruction
        question_start = range(R.first_text + 1, R.first_text + 1 + len(instruction))
        block = lambda heads, positions=instruction: Patch().block(heads, later, positions, layers=range(n_layers(model)) if heads is None else None)
        patches = {"none": Patch(), "every_head": block(None), "instruction_heads": block(heads),
                   "every_head_except_these": block(others), "instruction_heads_other_question_tokens": block(heads, question_start)}
        for draw in range(3):
            patches[f"random_heads_{draw}"] = block(random_like(heads, seed=draw))
        answers = run_patches(model, R, list(patches.values()))["answers"]
        results.add(index, **{name: answer == D.answer for name, answer in zip(patches, answers)})
    for name, paper in (("none", 0.0), ("every_head", 100.0), ("instruction_heads", 98.4), ("every_head_except_these", 0.0),
                        ("random_heads_0", "at most 1.6"), ("random_heads_1", "at most 1.6"), ("random_heads_2", "at most 1.6"),
                        ("instruction_heads_other_question_tokens", 0.0)):
        results.report(name, paper=paper, label=f"blocked: {name}, gives the left-to-right answer")
    results.save()


if __name__ == "__main__":
    main()
