"""Residual patch at the colon token, one layer at a time (Section 4, Figure 2a).

We copy the residual vector of the colon token from the source run into the target run, at one
layer, and read the target's next token. Each outcome letter tells what the patch moved:
T nothing, K the pointer, D the answer itself, P the request, R the layout. A patch at layer L
replaces the input to layer L, and the last layer number is the input to the final normalization.

    python binding/main_experiments/01_residual_patch_by_layer.py --num_pairs 500

Paper: the target keeps T in 99.6% of cases or more up to layer 20. It answers K at layers 23-29
(78.6% at layer 24). From layer 30 on it answers D in 95.4% or more. P and R stay at 6.2% or less.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs, outcome_of
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import make_patch, run_patches

# The cells the paper quotes, as (outcome, layer): percent of cases with that outcome.
PAPER = {("T", 19): 99.6, ("K", 23): 73.4, ("K", 24): 78.6, ("K", 29): 33.4, ("D", 30): 95.4, ("P", 26): 6.2}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    layers = range(n_layers(model) + 1)
    results = Results("residual_patch_by_layer", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, letters in both_directions(model, pair):
            patches = [make_patch(source, target, "residual", "colon", layer=layer) for layer in layers]
            answers = run_patches(model, target, patches)["answers"]
            outcomes = "".join(outcome_of(answer, letters) for answer in answers)     # one character per layer
            cells = {f"{outcome}_layer{layer}": outcomes[layer] == outcome for outcome, layer in PAPER if layer in layers}
            results.add(pair["index"], outcomes=outcomes, **cells)

    print("percent of cases with each outcome\nlayer" + "".join(f"{outcome:>7s}" for outcome in "TKDPR"))
    for layer in layers:
        column = [row["outcomes"][layer] for row in results.rows]
        print(f"{layer:5d}" + "".join(f"{100 * column.count(outcome) / len(column):7.1f}" for outcome in "TKDPR"))
    for (outcome, layer), paper in PAPER.items():
        results.report(f"{outcome}_layer{layer}", paper=paper, label=f"{outcome} after the patch at layer {layer}")
    results.save()


if __name__ == "__main__":
    main()
