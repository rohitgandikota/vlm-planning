"""Residual patch at every text token and every layer (Section 4.1, Figure 2b).

The same patch as 01_residual_patch_by_layer.py, now at each of the 22 tokens after the image.
We count the cases in which the target answers K, its own letter at the panel the source points
to. This shows which tokens carry the pointer, and at which layers.

    python binding/main_experiments/02_residual_patch_every_token.py --num_pairs 500

Paper: the pointer shows at the shape token at layers 16-20 (36.1% at layer 20) and at the colon
token at layers 23-29 (78.6% at layer 24). No other token reaches 5% at any layer.
The full token-by-layer map is saved in the result file under "map".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs, outcome_of
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import make_patch, run_patches

# The cells the paper quotes, as (token, layer): percent of cases that answer K.
PAPER = {("shape", 16): 13.3, ("shape", 17): 6.9, ("shape", 18): 6.0, ("shape", 19): 17.2, ("shape", 20): 36.1,
         ("colon", 23): 73.4, ("colon", 24): 78.6, ("colon", 25): 52.8, ("colon", 26): 62.0, ("colon", 27): 54.8,
         ("colon", 28): 51.6, ("colon", 29): 33.4}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    layers = range(n_layers(model) + 1)
    results = Results("residual_patch_every_token", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, letters in both_directions(model, pair):
            tokens = target.text_tokens()                           # (position, text) of every token after the image
            patches = [make_patch(source, target, "residual", position, layer=layer)
                       for position, _ in tokens for layer in layers]
            answers = run_patches(model, target, patches)["answers"]
            outcomes = "".join(outcome_of(answer, letters) for answer in answers)
            grid = [outcomes[i:i + len(layers)] for i in range(0, len(outcomes), len(layers))]    # one string per token
            first = target.first_text
            cells = {f"{name}_layer{layer}": grid[target.token(name) - first][layer] == "K"
                     for name, layer in PAPER if layer in layers}
            results.add(pair["index"], grid=grid, **cells)

    # The map: percent of cases that answer K, for every token (rows) and layer (columns). The token
    # texts are the same in every case, except for the shape token.
    labels = ["shape" if position == target.token("shape") else text for position, text in tokens]
    rows = results.rows
    k_map = [[100 * sum(row["grid"][t][layer] == "K" for row in rows) / len(rows) for layer in layers]
             for t in range(len(labels))]
    shape, colon = labels.index("shape"), len(labels) - 1
    print("percent of cases that answer K\nlayer   shape token   colon token   largest other token")
    for layer in layers:
        other = max(k_map[t][layer] for t in range(len(labels)) if t not in (shape, colon))
        print(f"{layer:5d} {k_map[shape][layer]:13.1f} {k_map[colon][layer]:13.1f} {other:21.1f}")
    for (name, layer), paper in PAPER.items():
        results.report(f"{name}_layer{layer}", paper=paper, label=f"K, patch at the {name} token, layer {layer}")
    # Appendix B.1 shows the same map for all five outcome letters.
    maps = {o: [[100 * sum(row["grid"][t][layer] == o for row in rows) / len(rows) for layer in layers]
                for t in range(len(labels))] for o in "TDKPR"}
    results.save(map={"tokens": labels, "percent_K": k_map, "percent": maps})


if __name__ == "__main__":
    main()
