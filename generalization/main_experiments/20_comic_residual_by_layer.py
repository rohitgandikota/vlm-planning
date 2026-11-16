"""On comic images, the colon token holds the pointer at the same layers (Appendix G.3.1, Figure 16).

The first version of the one-answer task: each panel holds a comic drawing under a white header that
carries the shape and its letter, letters come from all 26 capitals, and the prompt is longer ("Each
panel has one shape marker containing one letter. What is the letter inside the {shape} shape?").
As in binding/main_experiments/01, we copy the residual vector of the colon token from the source
into the target at one layer and read the next token (T, K, D, P, R as there). With no head patched,
we also read which panel the published Gaze Heads look at from the colon token.

    python generalization/main_experiments/20_comic_residual_by_layer.py --comics /path/to/comics

The comic drawings are not part of this repository: --comics must name a folder with 500 four-panel
comics as comic1/p1.png ... comic1/p4.png up to comic500/p4.png (see utils.dataset.comic_strip).
The 250 comic pairs are fixed by a seed; the paper ran this sweep on all 250 pairs (both
directions, 500 cases), with no no-patch filter. --num_pairs counts pairs from pair 0.

Paper: the pointer K peaks at 80.2% at layer 24, and the source's letter D comes through from layer
30. At layer 23 the published Gaze Heads look at the source's panel in 45.0% of cases.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import comic_both_directions, comic_pair, folder_argument, outcome_of
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model, n_layers
from utils.patching import make_patch, panel_attention, run_patches


def main():
    comics = folder_argument("--comics", "a folder with comic1/p1.png ... comic500/p4.png")
    args = parse_args(__doc__, num_pairs=250)
    model = load_model(args.model, args.device)
    layers = range(n_layers(model) + 1)
    results = Results("comic_residual_by_layer", args)
    for pair in [comic_pair(i) for i in range(args.start, min(250, args.start + args.num_pairs))]:
        for source, target, letters in comic_both_directions(model, comics, pair):
            patches = [make_patch(source, target, "residual", "colon", layer=layer) for layer in layers]
            result = run_patches(model, target, patches, keep_attention=True)
            outcomes = "".join(outcome_of(answer, letters) for answer in result["answers"])
            looked_at = panel_attention(result["attention"], HEADS["published_gaze"], target).argmax(dim=1).tolist()
            source_panel = target.image.letters.index(letters["K"])          # the panel the source points to
            results.add(pair["index"], outcomes=outcomes, **{f"K_layer{layer}": outcomes[layer] == "K" for layer in layers},
                        **{f"gaze_on_source_panel_layer{layer}": panel == source_panel for layer, panel in zip(layers, looked_at)})
    print("percent of cases with each outcome\nlayer" + "".join(f"{outcome:>7s}" for outcome in "TKDPR"))
    for layer in layers:
        column = [row["outcomes"][layer] for row in results.rows]
        print(f"{layer:5d}" + "".join(f"{100 * column.count(outcome) / len(column):7.1f}" for outcome in "TKDPR"))
    results.report("K_layer24", paper=80.2, label="K after the patch at layer 24")
    results.report("gaze_on_source_panel_layer23", paper=45.0, label="layer 23: Gaze Heads look at the source's panel")
    results.save()


if __name__ == "__main__":
    main()
