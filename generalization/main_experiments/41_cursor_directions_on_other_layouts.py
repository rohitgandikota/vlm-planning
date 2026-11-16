"""Every layout has a cursor, with its own code (Appendix G.3.8).

Appendix D writes the cursor on strips with one direction per panel. Here we fit such directions on
four layouts (the row of panels, a 2 x 3 grid, a column, and shapes on a blank canvas, numbered in
reading order) and ask whether the row's directions also work on the other layouts.
    Fit, per layout, on 60 images with the asked shape at four of six places. For every matching
        place p we write the right letters up to p and record the residual of the last token (the
        comma token, or the colon token when nothing is written) entering layers 20 and 21; a run
        is kept when its next letter is right. d_p is the mean residual of the runs "done up to p"
        minus the mean over p = 1..6 of these means (the grand mean).
    Test, per layout, on 100 new images with nothing written. At the colon token we set the residual
        to the grand mean plus d_p, for p = 2 and 3, with the directions of the row or of the layout
        itself. If the directions write the cursor, the first answer is the first matching place
        after p. Control: the grand mean plus a random direction of the same norm (five draws).
We also print the cosine between the row's d_p and each layout's d_p.

    python generalization/main_experiments/41_cursor_directions_on_other_layouts.py

--num_pairs is the number of test images per layout (the paper used 100; 60 fit images are used).
A test image serves p when its own first answer is at or before p and a matching place lies after p.

Paper: the row's directions do not carry over to the grid or the canvas, and only partly to the
column. Directions fitted on each layout work there: grid 98.8%, column 95.0%, canvas 81.3% for
p = 3, against 36.2% to 63.8% for random directions. Cosines with the row's directions: 0.18 to 0.93.
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import LETTERS, LIST_ASK, list_layout_side, make_variant, run_list_variant, sample_ordered_centers
from utils.experiment import Results, parse_args
from utils.model import load_model
from utils.patching import Patch, run_patches

LAYOUTS = ("row", "grid", "column", "canvas")
LAYERS = (20, 21)


def sample_image(rng, layout):
    side = list_layout_side(rng, rng.choice(LIST_ASK), sorted(rng.sample(range(6), 4)), [""] * 6)
    side["letters"] = rng.sample(LETTERS, 6)
    if layout == "canvas":
        side["canvas_centers"] = sample_ordered_centers(rng, 6, 960, [side["matching"]])
    return side


def fit_directions(model, layout, count, seed):
    """{layer: {p: d_p}} and {layer: grand mean}, from `count` images of one layout."""
    variant, rng = make_variant(layout), random.Random(seed)
    kept = {layer: {p: [] for p in range(7)} for layer in LAYERS}
    for _ in range(count):
        side = sample_image(rng, layout)
        letters, places = side["letters"], side["matching"]
        for p in [0] + [place + 1 for place in places]:                        # p = places done, 0 = nothing written
            after = [place for place in places if place > p - 1]
            run = run_list_variant(model, side, variant, [letters[place] for place in places if place < p])
            if (run.answer == " " + letters[after[0]]) if after else p == 6:
                for layer in LAYERS:
                    kept[layer][p].append(run.residual(layer, "last").float().cpu())
    directions, grand = {}, {}
    for layer in LAYERS:
        means = {p: torch.stack(vectors).mean(0) for p, vectors in kept[layer].items() if p > 0 and vectors}
        grand[layer] = torch.stack(list(means.values())).mean(0)
        directions[layer] = {p: mean - grand[layer] for p, mean in means.items()}
    return directions, grand


def main():
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    fitted = {layout: fit_directions(model, layout, 60, 10001 + i) for i, layout in enumerate(LAYOUTS)}
    results = Results("cursor_directions_on_other_layouts", args)
    for i, layout in enumerate(LAYOUTS):
        for layer in LAYERS:
            cosines = [float(torch.cosine_similarity(fitted["row"][0][layer][p], fitted[layout][0][layer][p], dim=0))
                       for p in range(1, 7) if p in fitted["row"][0][layer] and p in fitted[layout][0][layer]]
            print(f"{layout}, layer {layer}: cosine of the row's d_p with this layout's d_p:", [round(c, 2) for c in cosines])
        variant, rng = make_variant(layout), random.Random(10002 + 10 * i)
        for index in range(args.num_pairs):
            side = sample_image(rng, layout)
            letters, places = side["letters"], side["matching"]
            target = run_list_variant(model, side, variant, [])
            if target.answer != " " + letters[places[0]]:
                continue
            row = {}
            for p in (2, 3):
                if not (places[0] <= p - 1 and any(place > p - 1 for place in places)):
                    continue
                success = " " + letters[next(place for place in places if place > p - 1)]
                patches, names = [], []
                for layer in LAYERS:
                    for source in sorted({"row", layout}):
                        directions, grand = fitted[source]
                        if p in directions[layer]:
                            patches.append(Patch().residual(layer, target.token("colon"), (grand[layer] + directions[layer][p]).to(model.device)))
                            names.append(f"{layout}_p{p}_layer{layer}_{'own' if source == layout else 'row'}_directions")
                    d = fitted[layout][0][layer].get(p)
                    for draw in range(5 if d is not None else 0):
                        noise = torch.randn(d.shape, generator=torch.Generator().manual_seed(draw))
                        patches.append(Patch().residual(layer, target.token("colon"),
                                                        (fitted[layout][1][layer] + noise / noise.norm() * d.norm()).to(model.device)))
                        names.append(f"{layout}_p{p}_layer{layer}_random_{draw}")
                for name, answer in zip(names, run_patches(model, target, patches)["answers"]):
                    row[name] = answer == success
            results.add(f"{layout}{index}", **row)
    for layout, paper in (("row", None), ("grid", 98.8), ("column", 95.0), ("canvas", 81.3)):
        for layer in LAYERS:
            results.report(f"{layout}_p3_layer{layer}_own_directions", paper=paper, label=f"{layout}, p = 3, layer {layer}: own directions")
            if layout != "row":
                results.report(f"{layout}_p3_layer{layer}_row_directions", label=f"{layout}, p = 3, layer {layer}: the row's directions")
            results.report(f"{layout}_p3_layer{layer}_random_0", paper="36.2 to 63.8", label=f"{layout}, p = 3, layer {layer}: random direction")
    results.save()


if __name__ == "__main__":
    main()
