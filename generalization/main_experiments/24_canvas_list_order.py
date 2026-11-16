"""In what order does the model list shapes on a blank canvas? (Appendix G.3.2, "Order of a list").

With no panels, "left to right" no longer defines the order of a list. We draw six shapes at random
places on one 960 x 960 white canvas, the asked shape at four of them, ask the paper's list question,
and let the model write its list. We compare the list with three orders of the four asked shapes:
    left to right       by the x of their centers
    reading order       rows from the top, left to right inside a row: take the topmost remaining
                        shape and everything within 160 pixels below it as a row
                        (utils.dataset.reading_order)
    the same, on clear layouts only: the layouts where the reading order is the same for row heights
                        of 60, 160 and 260 pixels, for the six shapes and every subset of the asked
                        ones. The list pairs of 27_list_task_on_other_images.py use only such layouts.
No patch is used.

    python generalization/main_experiments/24_canvas_list_order.py

--num_pairs is the number of images (the paper used 60).

Paper: the model names the right set of letters in 98.3% of images, in left-to-right order in only
8.3%. The reading-order rule reproduces 79.7% of the lists with the right set, and 100.0% on the
clear layouts.
"""
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import (LETTERS, LIST_ASK, SHAPES, clear_order, draw_canvas, list_layout_side, list_question,
                           reading_order, sample_centers)
from utils.experiment import Results, parse_args
from utils.model import load_model
from utils.patching import run_model, write_tokens

SEED = 20260920


def main():
    args = parse_args(__doc__, num_pairs=60)
    model = load_model(args.model, args.device)
    results = Results("canvas_list_order", args)
    rng = random.Random(SEED)
    for index in range(args.start + args.num_pairs):
        ask = LIST_ASK[index % 3]
        side = list_layout_side(rng, ask, sorted(rng.sample(range(6), 4)), [""] * 6, fillers=SHAPES)
        side["letters"] = rng.sample(LETTERS, 6)
        centers = sample_centers(rng, 6, 960)
        rng.sample(SHAPES, 4), rng.sample(LETTERS, 4), sample_centers(rng, 4, 768)   # the paper's run drew a one-answer image here
        if index < args.start:                                    # keep the random draws of earlier images
            continue
        run = run_model(model, draw_canvas(side["shapes"], side["letters"], centers), list_question(ask), record=False)
        written = [item.strip() for item in write_tokens(model, run, max_tokens=24).split(",") if item.strip()]
        letters, places = side["letters"], side["matching"]
        in_order = lambda order: written == [letters[p] for p in order]
        right_set = sorted(written) == sorted(letters[p] for p in places)
        clear = clear_order(centers, range(6)) is not None and all(
            clear_order(centers, list(group)) is not None for k in range(2, 5) for group in itertools.combinations(places, k))
        results.add(index, right_set=right_set, left_to_right=in_order(sorted(places, key=lambda p: centers[p][0])),
                    reading_order=in_order(reading_order(centers, places)) if right_set else None,
                    reading_order_clear_layouts=in_order(reading_order(centers, places)) if right_set and clear else None,
                    written=written, centers=centers, places=places)
    results.report("right_set", paper=98.3, label="right set of letters")
    results.report("left_to_right", paper=8.3, label="in left-to-right order")
    results.report("reading_order", paper=79.7, label="reading-order rule (lists with the right set)")
    results.report("reading_order_clear_layouts", paper=100.0, label="reading-order rule, clear layouts")
    results.save()


if __name__ == "__main__":
    main()
