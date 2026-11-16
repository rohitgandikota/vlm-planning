"""Find the heads that read "List them from right to left." (Appendix G.3.5, "The heads that read the instruction").

Images and runs as in 34_right_to_left_instruction_blocked.py: R is the run with the right-to-left
sentence, D the run with the paper's prompt, on the same row image with the same written letter.
Here the letter of r2 is written, so R should answer r3 and D answers something else. The edit: in
a set of heads, every token after the instruction cannot attend to the tokens of the instruction.
    1. Search images. For every head, two edits: only that head blocked, and every head blocked
       except that head. With LD = logit of R's answer minus logit of D's answer, the effect of an
       edit is (LD unpatched - LD patched) / (LD unpatched - LD with every head blocked). The score of
       a head is the mean of its own effect and the share it keeps alone (1 - effect of the second edit).
    2. Validation images. Block the top K heads, K = 1, 2, 4, ..., 128, and every head. The number of
       heads is the smallest K whose rate of D's answer reaches 90% of the rate with every head blocked.
The held-out test of the chosen heads is 35_instruction_heads_blocked.py (HEADS["instruction"]).

    python generalization/main_experiments/00_find_instruction_heads.py

--num_pairs is the number of search images (the paper used 48), and 60 validation images are used.
This is a head search over 1,152 heads, twice per image, and is slow.

Paper: the 16 heads in layers 18-23 listed in HEADS["instruction"], led by L18H17.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import instruction_positions, right_to_left_items
from utils.experiment import Results, parse_args
from utils.heads import HEADS, rank_heads
from utils.model import all_heads, load_model, n_heads, n_layers
from utils.patching import Patch, run_patches

SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


def blocked(model, run, heads):
    """`heads` (None: every head of every layer) cannot attend from the tokens after the instruction to it."""
    instruction = instruction_positions(run)
    later = range(instruction[-1] + 1, run.last + 1)
    return Patch().block(heads, later, instruction, layers=range(n_layers(model)) if heads is None else None)


def main():
    args = parse_args(__doc__, split="search", num_pairs=48)
    model = load_model(args.model, args.device)
    every_head = all_heads(model)
    score, used = torch.zeros(n_layers(model), n_heads(model)), 0
    for R, D in right_to_left_items(model, 980, args.num_pairs):
        margin = lambda logits: logits[..., R.answer_id] - logits[..., D.answer_id]
        unpatched = float(margin(R.logits.cpu()))
        full = float(margin(run_patches(model, R, [blocked(model, R, None)])["logits"][0]))
        if unpatched - full <= 0:
            continue
        alone = run_patches(model, R, [blocked(model, R, [head]) for head in every_head], max_batch=32)["logits"]
        kept = run_patches(model, R, [blocked(model, R, [h for h in every_head if h != head]) for head in every_head], max_batch=32)["logits"]
        cut_effect = (unpatched - margin(alone)) / (unpatched - full)
        keep_share = 1 - (unpatched - margin(kept)) / (unpatched - full)
        score += ((cut_effect + keep_share) / 2).view(n_layers(model), n_heads(model))
        used += 1
        print(f"search image {used} scored", flush=True)
    ranking = rank_heads(score / max(used, 1))
    results = Results("find_instruction_heads", args)
    for index, (R, D) in enumerate(right_to_left_items(model, 984, 60)):
        answers = run_patches(model, R, [blocked(model, R, ranking[:k]) for k in SIZES] + [blocked(model, R, None)])["answers"]
        results.add(index, **{f"top_{k}": answer == D.answer for k, answer in zip(SIZES, answers)}, every_head=answers[-1] == D.answer)
    ceiling = results.report("every_head", label="every head blocked: gives D's answer")["percent"]
    rates = {k: results.report(f"top_{k}", label=f"top {k} heads blocked: gives D's answer")["percent"] for k in SIZES}
    chosen = next((k for k in SIZES if rates[k] >= 0.9 * ceiling), n_layers(model) * n_heads(model))
    print(f"\nchosen K = {chosen}: {ranking[:chosen]}")
    print(f"{len(set(ranking[:chosen]) & set(HEADS['instruction']))} of them are among the paper's 16 instruction heads")
    results.save(found=ranking[:chosen], ranking=ranking[:256], scores=(score / max(used, 1)).tolist())


if __name__ == "__main__":
    main()
