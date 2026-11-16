"""The comma token holds the matching panels and the next panel (Section 5.4).

No patch. The target has written one letter. We record the residual vector of its last comma token
at every layer and fit two linear probes per layer:
    matching panels   six outputs, one per panel: does this panel hold the shape we ask for?
                      A panel is read correctly when its output is on the right side of 0.5.
    next panel        which of the six panels the next answer comes from (the largest output).
The probes are fitted on the first half of the pairs and tested on the second half. Appendix E.4
adds a third probe, for which panel was written (the first matching panel).

    python progress/main_experiments/12_probes_at_the_comma_token.py --num_pairs 500

Paper: the matching panels are read with 95.9% accuracy per panel at layer 17 and 99.3% at layer
21. The next panel is read with 98.0% at layer 18 and 100.0% at layer 20. Appendix E.4: the
written panel is read with 94.0% at layer 12 and 100.0% at layer 18. For these numbers the
paper drew the target images again with every shape at its panel center, and stored the residual
vectors in half precision. This file does the same. Every held-out pair is used.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import fit_linear_probe


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("probes_at_the_comma_token", args)
    pairs = list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]
    layers = range(n_layers(model) + 1)                            # the last one is after the last layer
    vectors = []                                                   # per pair: the comma token's residual at every layer
    for pair in pairs:
        centered = {**pair["target"], "centers": None}             # the same image with every shape at its panel center
        run = run_list(model, centered, [pair["written"]])
        vectors.append(torch.stack([run.residual(layer, "last_comma") for layer in layers]).half().cpu().numpy())
    vectors = np.stack(vectors)                                    # (pairs, layers, hidden size)
    matching = np.array([[panel in pair["panels"] for panel in range(6)] for pair in pairs], float)
    next_panel = np.array([pair["panels"][1] for pair in pairs])   # one letter is written: the second matching panel is next
    half = len(pairs) // 2
    written_panel = np.array([pair["panels"][0] for pair in pairs])
    panel_right, next_right, written_right = {}, {}, {}            # per layer: which test cases the probes read correctly
    for layer in layers:
        fit, test = vectors[:half, layer], vectors[half:, layer]
        panel_right[layer] = (fit_linear_probe(fit, matching[:half])(test) > 0.5) == (matching[half:] > 0.5)
        next_right[layer] = fit_linear_probe(fit, np.eye(6)[next_panel[:half]])(test).argmax(1) == next_panel[half:]
        written_right[layer] = fit_linear_probe(fit, np.eye(6)[written_panel[:half]])(test).argmax(1) == written_panel[half:]
    for i, pair in enumerate(pairs[half:]):
        results.add(pair["index"], next_panel_layer_18=bool(next_right[18][i]), next_panel_layer_20=bool(next_right[20][i]),
                    written_panel_layer_12=bool(written_right[12][i]), written_panel_layer_18=bool(written_right[18][i]))
        for panel in range(6):                                     # one case per panel
            results.add(pair["index"], matching_panels_layer_17=bool(panel_right[17][i, panel]),
                        matching_panels_layer_21=bool(panel_right[21][i, panel]))
    results.report("matching_panels_layer_17", paper=95.9, label="matching panels, layer 17 (per panel)")
    results.report("matching_panels_layer_21", paper=99.3, label="matching panels, layer 21 (per panel)")
    results.report("next_panel_layer_18", paper=98.0, label="next panel, layer 18")
    results.report("next_panel_layer_20", paper=100.0, label="next panel, layer 20")
    results.report("written_panel_layer_12", paper=94.0, label="written panel, layer 12 (Appendix E.4)")
    results.report("written_panel_layer_18", paper=100.0, label="written panel, layer 18 (Appendix E.4)")
    results.save(accuracy_by_layer={"matching_panels": [100 * float(panel_right[layer].mean()) for layer in layers],
                                    "next_panel": [100 * float(next_right[layer].mean()) for layer in layers],
                                    "written_panel": [100 * float(written_right[layer].mean()) for layer in layers]})


if __name__ == "__main__":
    main()
