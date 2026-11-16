"""Holding the comma token's residual, with the cursor moved by the binding heads (Appendix E.4).

This extends progress/main_experiments/14_holding_the_comma_token.py, which moves the cursor with
the 32 answer-token heads. Here the 16 binding heads with the query and key heads (18 heads) read
the source's image keys at the answer token of W, which sits at r2 in the source image, so the
target answers from r3 instead of r2. On top of this patch we hold the residual of the last comma
token entering one layer (14 to 32) at the target's own unpatched values. As a control we hold the
residual alone, with no patch, at layers 16, 20, 24, 28, 32.

    python progress/appendix_experiments/52_holding_the_comma_token_binding_heads.py --num_pairs 500

Paper: the binding heads alone move the answer to r3 in 48.2% of cases. With the residual held at
layer 23 the target's own answer comes back in 99.0%. The hold alone keeps the target's own answer
in 100.0% of cases at every layer tried. A pair counts when both runs give their own next letter
with no patch (500 of 500).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import Patch, make_patch, run_patches

HELD_LAYERS = range(14, 33)
CONTROL_LAYERS = (16, 20, 24, 28, 32)


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("holding_the_comma_token_binding_heads", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the target's letters at r1, r2, r3, r4
        target = run_list(model, pair["target"], [pair["written"]])
        source = run_list(model, pair["source"], [pair["written"]])
        if target.answer != " " + letters[1] or source.answer != " " + matching_letters(pair["source"])[1]:
            continue
        move = lambda: make_patch(source, target, "image_keys", "last_answer", heads=HEADS["binding_query_key"])
        hold = lambda layer: make_patch(target, target, "residual", "last_comma", layer=layer)
        patches = {"alone": move()}
        patches.update({layer: move().add(hold(layer)) for layer in HELD_LAYERS})
        patches.update({f"hold_alone_{layer}": Patch().add(hold(layer)) for layer in CONTROL_LAYERS})
        answers = dict(zip(patches, run_patches(model, target, list(patches.values()))["answers"]))
        own, moved = " " + letters[1], " " + letters[2]            # the target's own answer, and the answer after r2
        results.add(pair["index"], moved_patch_alone=answers["alone"] == moved,
                    **{f"own_held_at_layer_{layer}": answers[layer] == own for layer in HELD_LAYERS},
                    **{f"moved_held_at_layer_{layer}": answers[layer] == moved for layer in HELD_LAYERS},
                    **{f"own_hold_alone_at_layer_{layer}": answers[f"hold_alone_{layer}"] == own for layer in CONTROL_LAYERS})
    results.report("moved_patch_alone", paper=48.2, label="binding heads move the cursor: answer moves")
    results.report("own_held_at_layer_23", paper=99.0, label="residual held at layer 23: own answer back")
    for layer in CONTROL_LAYERS:
        results.report(f"own_hold_alone_at_layer_{layer}", paper=100.0, label=f"hold alone at layer {layer}: own answer")
    results.save()


if __name__ == "__main__":
    main()
