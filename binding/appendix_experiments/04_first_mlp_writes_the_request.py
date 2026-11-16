"""The first MLP writes the request (Appendix B.3, Figure "mlp-sweep").

The query heads carry the request, so something must write it into the shape token before the
binding layers. We copy the output of one MLP at the shape token from the source run into the
target run, for each of the 36 layers in turn. If that MLP writes the request, the target answers
the source's question on its own image: its letter inside the shape the source asks for (P). We
also count K, the pointer.

    python binding/appendix_experiments/04_first_mlp_writes_the_request.py --num_pairs 500

Paper: the first MLP (layer 0) moves the request in 98.3% of cases. No other MLP does so in more
than 6.1% (layer 1). The table of P and K per layer is saved under "by_layer".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs, outcome_of
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    layers = range(n_layers(model))
    results = Results("first_mlp_writes_the_request", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, letters in both_directions(model, pair):
            patches = [make_patch(source, target, "mlp", "shape", layer=layer) for layer in layers]
            outcomes = "".join(outcome_of(answer, letters) for answer in run_patches(model, target, patches)["answers"])
            results.add(pair["index"], outcomes=outcomes, first_mlp=outcomes[0] == "P",
                        **{f"mlp{layer}": outcomes[layer] == "P" for layer in layers if layer > 0})

    by_layer = {"P": [], "K": []}
    print("percent of cases\nlayer      P      K")
    for layer in layers:
        column = [row["outcomes"][layer] for row in results.rows]
        for outcome in by_layer:
            by_layer[outcome].append(100 * column.count(outcome) / len(column))
        print(f"{layer:5d} {by_layer['P'][layer]:6.1f} {by_layer['K'][layer]:6.1f}")
    results.report("first_mlp", paper=98.3, label="first MLP: request moved (P)")
    largest = max(layers[1:], key=lambda layer: by_layer["P"][layer])
    results.report(f"mlp{largest}", paper=6.1, label=f"largest other MLP (layer {largest}): request moved (P)")
    results.save(by_layer=by_layer)


if __name__ == "__main__":
    main()
