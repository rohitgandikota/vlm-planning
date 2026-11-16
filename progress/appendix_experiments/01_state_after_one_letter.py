"""The early state says "one letter was written", whichever letter it was (Appendix D.2, figure panel a).

Before the cursor forms, what does the comma token hold? We use one image twice. The target has
written nothing, so its next answer is its letter at r1. The source is the same image with one
letter written, either r1 or r2. We patch the residual vector of the source's comma token into the
target's colon token, one layer at a time from 14 to 32, and read which matching panel the
target's next letter comes from.
    r1 written   r2 means the target skipped the written panel
    r2 written   r2 means the state says only "one letter is done" (it skips one panel, the first);
                 r3 means the state says which panel was written (the target starts after r2)

    python progress/appendix_experiments/01_state_after_one_letter.py --num_pairs 500

Paper: with r1 written, r2 in 65.8% of cases at layer 14 and 96.4% at layer 21. With r2 written, r2
in 75.1% at layer 19 (r3 in 2.1%), and r3 in 82.4% at layer 23 and 92.7% at layer 32.
These numbers came from an earlier, smaller run: 200 list pairs drawn from another seed, of which
193 count. Every held-out pair here counts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, name_answer, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model
from utils.patching import make_patch, run_patches

LAYERS = range(14, 33)


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("state_after_one_letter", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the target's letters at r1 to r4
        names = {f"r{i + 1}": letter for i, letter in enumerate(letters)}
        target = run_list(model, pair["target"], [])
        row = {}
        for written in ("r1", "r2"):
            source = run_list(model, pair["target"], [names[written]])      # the same image, one letter written
            patches = [make_patch(source, target, "residual", "colon", source_token="last_comma", layer=layer)
                       for layer in LAYERS]
            said = [name_answer(answer, names) for answer in run_patches(model, target, patches)["answers"]]
            row[f"{written}_written"] = said                             # saved with the row, not counted
            for layer, panel in zip(LAYERS, said):
                for outcome in ("r2", "r3"):
                    row[f"{written}_written_layer_{layer}_says_{outcome}"] = panel == outcome
        results.add(pair["index"], **row)
    results.report("r1_written_layer_14_says_r2", paper=65.8, label="r1 written, layer 14: answers r2")
    results.report("r1_written_layer_21_says_r2", paper=96.4, label="r1 written, layer 21: answers r2")
    results.report("r2_written_layer_19_says_r2", paper=75.1, label="r2 written, layer 19: answers r2")
    results.report("r2_written_layer_19_says_r3", paper=2.1, label="r2 written, layer 19: answers r3")
    results.report("r2_written_layer_23_says_r3", paper=82.4, label="r2 written, layer 23: answers r3")
    results.report("r2_written_layer_32_says_r3", paper=92.7, label="r2 written, layer 32: answers r3")
    results.save()


if __name__ == "__main__":
    main()
