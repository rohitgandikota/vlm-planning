"""The comma token finds the matching panels again by itself (Appendix E.3, "The next answer").

This extends progress/main_experiments/13_matching_panels_patched.py. The target has its shape at
r1 to r4 and has written W, its letter at r1. The source "source_set" has the shape at r1 and r3
only and has written the same W, so only its set of matching panels differs. If the target takes
the source's set, it skips r2 and answers its letter at r3. Patches:
    shape-token residual      the source's residual at the shape token, entering layer 20
    every patch together      the outputs of every head at the shape token, every head except the
                              gaze heads reads the source's image keys at the comma token, every
                              head reads them at the answer token, and the comma token keeps only
                              the gaze heads' view of the image
and two blocks in the target's own run, with no source:
    comma token keeps only the gaze heads' view of the image
    every later token cannot attend to the shape token

    python progress/appendix_experiments/49_next_answer_found_again.py --num_pairs 500

Paper: the shape-token residual moves the answer to r3 in 14.8% of cases. With every patch
together it does so in 96.8%. In the target's own run the two blocks keep 91.8% and 66.5% of
answers right. A pair counts when both runs give their own next letter with no patch (499 of 500).
The appendix text describes the 96.8% patch as the 32 shape-token heads; the run that produced it
copied the outputs of every head at the shape token, and this file does the same.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import all_heads, load_model, n_layers
from utils.patching import Patch, make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    every_layer, every_head = range(n_layers(model)), all_heads(model)
    not_gaze = [head for head in every_head if head not in set(HEADS["gaze"])]
    results = Results("next_answer_found_again", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the target's letters at r1, r2, r3, r4
        target = run_list(model, pair["target"], [pair["written"]])
        source = run_list(model, pair["source_set"], [pair["written"]])
        if target.answer != " " + letters[1] or source.answer != " " + matching_letters(pair["source_set"])[1]:
            continue
        comma, shape, image = target.token("last_comma"), target.token("shape"), target.image_positions
        gaze_view_only = lambda: Patch().block(not_gaze, [comma], image)        # the comma sees the image through the gaze heads only
        together = make_patch(source, target, "head_output", "shape", heads=every_head)
        together.add(make_patch(source, target, "image_keys", "last_comma", heads=not_gaze))
        together.add(make_patch(source, target, "image_keys", "last_answer", heads=every_head))
        patches = {
            "shape_token_residual": make_patch(source, target, "residual", "shape", layer=20),
            "every_patch_together": together.add(gaze_view_only()),
            "own_run_comma_keeps_only_gaze_view": gaze_view_only(),
            "own_run_later_tokens_cannot_read_shape_token": Patch().block(None, range(shape + 1, comma + 1), [shape], layers=every_layer),
        }
        answers = dict(zip(patches, run_patches(model, target, list(patches.values()))["answers"]))
        results.add(pair["index"], shape_token_residual=answers["shape_token_residual"] == " " + letters[2],
                    every_patch_together=answers["every_patch_together"] == " " + letters[2],
                    **{name: answers[name] == " " + letters[1] for name in patches if name.startswith("own_run")})
    results.report("shape_token_residual", paper=14.8, label="shape-token residual: answer moves to r3")
    results.report("every_patch_together", paper=96.8, label="every patch together: answer moves to r3")
    results.report("own_run_comma_keeps_only_gaze_view", paper=91.8, label="own run, comma sees through gaze heads: right")
    results.report("own_run_later_tokens_cannot_read_shape_token", paper=66.5, label="own run, shape token blocked: right")
    results.save()


if __name__ == "__main__":
    main()
