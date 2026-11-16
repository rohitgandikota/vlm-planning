"""The cursor acts through the comma token's residual (Section 5.4).

The target has its shape at four panels, r1 to r4, and has written W, its letter at r1. In the
source image, W sits at r2. We first move the cursor: the 32 answer-token heads at the answer
token read the source's image keys, so the target binds W to r2 and answers from r3 instead of r2.
On top of this patch we keep one part of the target at the values of its own unpatched run:
    the residual vector of the last comma token entering one layer (layers 14 to 32, one at a time)
    the queries of the 32 gaze heads at the last comma token
If the moved cursor reaches the answer through that part, the target's own answer comes back.

    python progress/main_experiments/14_holding_the_comma_token.py --num_pairs 500

Paper: the patch alone moves the answer to r3 in 73.8% of cases. With the residual held at layer
19 the patch keeps its effect in 69.8%. Held at layer 23 the target's own answer comes back in
99.4% (99.6% or more at every later layer). With the gaze heads' queries held it comes back in 99.8%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import make_patch, run_patches

HELD_LAYERS = range(14, 33)


def move_cursor(source, target):
    """The answer-token heads at the last answer token read the source's image keys."""
    return make_patch(source, target, "image_keys", "last_answer", heads=HEADS["answer_token"])


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("holding_the_comma_token", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the target's letters at r1, r2, r3, r4
        target = run_list(model, pair["target"], [pair["written"]])
        source = run_list(model, pair["source"], [pair["written"]])
        if target.answer != " " + letters[1] or source.answer != " " + matching_letters(pair["source"])[1]:
            continue
        hold_gaze_queries = make_patch(target, target, "query", "last_comma", heads=HEADS["gaze"])
        patches = {"alone": move_cursor(source, target), "gaze": move_cursor(source, target).add(hold_gaze_queries)}
        for layer in HELD_LAYERS:
            hold_residual = make_patch(target, target, "residual", "last_comma", layer=layer)
            patches[layer] = move_cursor(source, target).add(hold_residual)
        answers = dict(zip(patches, run_patches(model, target, list(patches.values()))["answers"]))
        own, moved = " " + letters[1], " " + letters[2]            # the target's own answer, and the answer after r2
        results.add(pair["index"], moved_patch_alone=answers["alone"] == moved, own_gaze_queries_held=answers["gaze"] == own,
                    **{f"moved_held_at_layer_{layer}": answers[layer] == moved for layer in HELD_LAYERS},
                    **{f"own_held_at_layer_{layer}": answers[layer] == own for layer in HELD_LAYERS})
    results.report("moved_patch_alone", paper=73.8, label="cursor moved, nothing held: answer moves")
    results.report("moved_held_at_layer_19", paper=69.8, label="residual held at layer 19: answer still moves")
    results.report("own_held_at_layer_23", paper=99.4, label="residual held at layer 23: own answer back")
    results.report("own_gaze_queries_held", paper=99.8, label="gaze-head queries held: own answer back")
    results.save()


if __name__ == "__main__":
    main()
