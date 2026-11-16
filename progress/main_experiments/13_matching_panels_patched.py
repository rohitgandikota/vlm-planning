"""The comma token finds the matching panels again by itself (Section 5.4).

The target has its shape at four panels, r1 to r4, and has written the letter at r1. The source
has the shape at r1 and r3 only and has written the same letter, so the two runs differ only in
their set of matching panels. We copy the outputs of the 32 heads found at the shape token of the
list task from the source into the target at the shape token. If the comma token took the matching
panels from the shape token alone, the target would now skip r2 and answer with its letter at r3.
We run the patch twice: alone, and with the comma token blocked from the image in every head
except the 32 gaze heads, so that it cannot look for the matching panels again.

    python progress/main_experiments/13_matching_panels_patched.py --num_pairs 500

Paper: the answer moves to r3 in 21.0% of cases with the patch alone, and in 64.5% when the comma
token keeps only the gaze heads' view of the image. Random heads at the shape token give 0.0%.
The source images exist for held-out pairs only. A pair counts when both runs give their own next
letter with no patch (499 of 500).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import all_heads, load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("matching_panels_patched", args)
    gaze = set(HEADS["gaze"])
    not_gaze = [head for head in all_heads(model) if head not in gaze]
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the target's letters at r1, r2, r3, r4
        target = run_list(model, pair["target"], [pair["written"]])
        source = run_list(model, pair["source_set"], [pair["written"]])
        if target.answer != " " + letters[1] or source.answer != " " + matching_letters(pair["source_set"])[1]:
            continue
        random_heads = random_like(HEADS["list_shape_token"], seed=pair["index"])
        alone = make_patch(source, target, "head_output", "shape", heads=HEADS["list_shape_token"])
        gaze_only = make_patch(source, target, "head_output", "shape", heads=HEADS["list_shape_token"])
        gaze_only.block(not_gaze, [target.token("last_comma")], target.image_positions)
        control = make_patch(source, target, "head_output", "shape", heads=random_heads)
        answers = run_patches(model, target, [alone, gaze_only, control])["answers"]
        moved = [answer == " " + letters[2] for answer in answers]  # the target answers from r3
        results.add(pair["index"], patch_alone=moved[0], comma_sees_image_through_gaze_heads_only=moved[1],
                    random_heads=moved[2], answer=answers[0])
    results.report("patch_alone", paper=21.0, label="matching panels patched: answer moves to r3")
    results.report("comma_sees_image_through_gaze_heads_only", paper=64.5, label="same, comma token keeps only the gaze heads' view")
    results.report("random_heads", paper=0.0, label="random heads at the shape token")
    results.save()


if __name__ == "__main__":
    main()
