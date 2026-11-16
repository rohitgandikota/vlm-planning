"""The cursor indexes separate images as it indexes panels (Appendix G.3.3).

Six separate 384 x 384 images in one user turn, each holding one shape and one letter, with the
paper's list question (utils.patching.run_model_images). Two parts:
    1. Lists. The asked shape is in 2 to 4 of the six images. Does the model list the letters
       across the images, in order?
    2. The cursor experiment of progress/main_experiments/03, with images in place of panels. The
       source asks for a shape in images 2, 4, 5 and has written images 2 and 4. The target asks for
       another shape in images 1, 2, 3, 6 and has written nothing. The source's last comma residual
       replaces the target's colon residual at one layer. A position after image 4 gives the target's
       letter in image 6; a count of two gives its third match (image 3). As a control, the source
       with nothing written gives its colon residual, which should leave the target's own first letter.

    python generalization/main_experiments/31_six_separate_images.py

--num_pairs is the number of lists in part 1 and of pairs in part 2 (the paper used 100 each). The
pairs of part 2 are drawn with utils.dataset.sample_position_pair from seed 1602 and kept when both
runs give their own next letter with no patch.

Paper: the model lists the letters in order in 97.0% of items, exactly in 92.0%. The target writes
the letter of its sixth image in 83.0% of pairs at layer 20 and 82.0% at layer 21; it holds the count
reading at layer 19 (87.0%), and from layer 30 it takes the source's letter. The source with nothing
written leaves the target's own first letter in 100.0% through layer 22.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import (LETTERS, LIST_ASK, list_layout_side, list_question, make_variant, name_answer,
                           run_list_variant, sample_position_pair, separate_images, variant_progress_pairs)
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import make_patch, run_model_images, run_patches, write_tokens


def lists(model, results, count):
    """Part 1: the model's own lists across six separate images."""
    rng = random.Random(1601)
    for index in range(count):
        ask = rng.choice(LIST_ASK)
        side = list_layout_side(rng, ask, sorted(rng.sample(range(6), rng.randint(2, 4))), [""] * 6)
        side["letters"] = rng.sample(LETTERS, 6)
        run = run_model_images(model, separate_images(side), list_question(ask), record=False)
        written = [item.strip() for item in write_tokens(model, run, max_tokens=16).split(",") if item.strip()]
        panels = [side["letters"].index(letter) for letter in written if letter in side["letters"]]
        results.add(f"list{index}", exact_list=written == [side["letters"][p] for p in side["matching"]],
                    in_order=panels == sorted(set(panels)), written=written)


def main():
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    results = Results("six_separate_images", args)
    lists(model, results, args.num_pairs)
    separate, layers = make_variant("separate"), range(n_layers(model) + 1)
    for pair in variant_progress_pairs(model, separate, sample_position_pair, seed=1602, num_pairs=args.num_pairs):
        source, target = run_list_variant(model, pair["source"], separate), run_list_variant(model, pair["target"], separate)
        nothing_written = run_list_variant(model, pair["source"], separate, written=[])
        patches = [make_patch(source, target, "residual", "colon", source_token="last_comma", layer=layer) for layer in layers]
        patches += [make_patch(nothing_written, target, "residual", "colon", layer=layer) for layer in layers]
        answers = run_patches(model, target, patches)["answers"]
        outcomes = [name_answer(answer, pair["letters"]) for answer in answers]
        written, control = outcomes[:len(layers)], outcomes[len(layers):]
        results.add(pair["index"], outcomes=written, control_outcomes=control,
                    **{f"sixth_image_layer_{layer}": outcome == "position" for layer, outcome in zip(layers, written)},
                    **{f"count_layer_{layer}": outcome == "count" for layer, outcome in zip(layers, written)},
                    **{f"nothing_written_own_layer_{layer}": outcome == "set" for layer, outcome in zip(layers, control)})
    results.report("in_order", paper=97.0, label="lists across the images in order")
    results.report("exact_list", paper=92.0, label="exact lists")
    results.report("sixth_image_layer_20", paper=83.0, label="layer 20: letter of the sixth image")
    results.report("sixth_image_layer_21", paper=82.0, label="layer 21: letter of the sixth image")
    results.report("count_layer_19", paper=87.0, label="layer 19: the count reading (third match)")
    results.report("nothing_written_own_layer_22", paper=100.0, label="nothing written, layer 22: own first letter")
    results.save()


if __name__ == "__main__":
    main()
