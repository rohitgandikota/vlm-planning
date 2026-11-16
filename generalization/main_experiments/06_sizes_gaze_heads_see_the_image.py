"""Do the gaze heads need to see the image, at every model size (Appendix G.1.3, Table "sizes-gaze", rows 4 and 5).

No source run. In the target's own run, with each size's own gaze heads:
    only the gaze heads see the image   every head at the tokens between the shape token and the
                                        colon token, and every other head at the colon token, is
                                        blocked from the image (as in gaze/main_experiments/03)
    gaze heads blocked                  the gaze heads at the colon token are blocked from the image
                                        (as in gaze/main_experiments/05)
Each counts the cases that keep the target's own answer T. The control of the first: random heads
from the same layers (seed = pair index) are the ones that still see the image.

    python generalization/main_experiments/06_sizes_gaze_heads_see_the_image.py --model Qwen/Qwen3-VL-2B-Instruct --num_pairs 500

Models: any Qwen3-VL size. 2B 16 GB, 4B 24 GB, 8B 48 GB, 32B 160 GB (one B200) or --device cpu.
3 patched runs per case, 2 cases per pair.

Paper (2B / 4B / 8B / 32B; the 2B and 4B on 100 earlier held-out pairs, the 32B on 20): only these
heads see the image 84.5 / 75.5 / 95.1 / 95.0; these heads blocked 25.5 / 95.5 / 88.2 / 100.0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, pairs_of_8b
from utils.experiment import Results, parse_args
from utils.heads import heads_of_size, random_heads_like
from utils.model import all_heads, load_model, n_layers
from utils.patching import Patch, run_patches

PAPER = {"only_gaze_heads_see": {"2B": 84.5, "4B": 75.5, "8B": 95.1, "32B": 95.0},
         "gaze_heads_blocked": {"2B": 25.5, "4B": 95.5, "8B": 88.2, "32B": 100.0}}


def only_these_see_the_image(model, target, heads):
    """Block the image for everything after the shape token, except `heads` at the colon token."""
    shape, colon = target.token("shape"), target.token("colon")
    patch = Patch().block(None, range(shape + 1, colon), target.image_positions, layers=range(n_layers(model)))
    return patch.block([head for head in all_heads(model) if head not in heads], [colon], target.image_positions)


def main():
    args = parse_args(__doc__)
    pairs = pairs_of_8b("one_answer", args.split, args.start + args.num_pairs, args.device)[args.start:]
    model = load_model(args.model, args.device)
    gaze = heads_of_size(args.model, "gaze")
    results = Results(f"sizes_gaze_heads_see_the_image_{args.model.split('/')[-1]}", args)
    for pair in pairs:
        random_heads = random_heads_like(model, gaze, seed=pair["index"])
        for _, target, letters in both_directions(model, pair):          # each side of the pair is a target once
            patches = {"only_gaze_heads_see": only_these_see_the_image(model, target, gaze),
                       "only_random_heads_see": only_these_see_the_image(model, target, random_heads),
                       "gaze_heads_blocked": Patch().block(gaze, [target.token("colon")], target.image_positions)}
            answers = run_patches(model, target, list(patches.values()))["answers"]
            results.add(pair["index"], **{name: answer == " " + letters["T"] for name, answer in zip(patches, answers)})
    size = args.model.split("-")[-2]
    results.report("only_gaze_heads_see", paper=PAPER["only_gaze_heads_see"].get(size), label="only the gaze heads see the image: T")
    results.report("only_random_heads_see", label="only random heads see the image: T")
    results.report("gaze_heads_blocked", paper=PAPER["gaze_heads_blocked"].get(size), label="gaze heads blocked from the image: T")
    results.save()


if __name__ == "__main__":
    main()
