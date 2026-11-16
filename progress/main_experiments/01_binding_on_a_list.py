"""Binding on a list (Section 5, opening).

When four panels hold the shape we ask for, do the binding heads find all of them? There is no
patch here. We run the target with nothing written and look at the shape token in two ways.

1. Attention. How much of the heads' image attention falls on the four matching panels?
2. Probe. We take what the heads write into the residual stream at the shape token. A linear probe
   for "panel p holds the shape" is fitted on the first half of the pairs and tested on the second.

The heads are the 16 binding heads and the 2 query and key heads outside them (18 heads).

    python progress/main_experiments/01_binding_on_a_list.py --num_pairs 500

Paper: 79.6% of the image attention is on the matching panels. The probe reads each panel in 97.7%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.patching import fit_linear_probe, heads_contribution
from utils.dataset import list_pairs, run_list
from utils.experiment import Results, interval, parse_args
from utils.heads import HEADS
from utils.model import load_model


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    heads = HEADS["binding_query_key"]
    results = Results("binding_on_a_list", args)
    pairs = list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]
    on_matching, on_image, vectors, holds_shape = [], [], [], []
    for pair in pairs:
        target = run_list(model, pair["target"])
        # 1. the heads' attention from the shape token to the image, summed over the heads
        attention = sum(target.attention(layer, head, "shape")[target.image_positions] for layer, head in heads)
        panels = target.panel_of_image_tokens()
        on_matching.append(sum(float(attention[panels == panel].sum()) for panel in pair["panels"]))
        on_image.append(float(attention.sum()))
        results.add(pair["index"], attention_on_matching=on_matching[-1], attention_on_image=on_image[-1])
        # 2. what the heads write at the shape token (kept in half precision, as in the paper's run)
        vectors.append(heads_contribution(model, target, heads, "shape").half().float().cpu().numpy())
        holds_shape.append([panel in pair["panels"] for panel in range(6)])

    # The share of attention is a ratio of sums over all pairs. The interval resamples pairs.
    low, high = interval(on_matching, on_image, [pair["index"] for pair in pairs])
    share = 100 * sum(on_matching) / sum(on_image)
    print(f"{'image attention on the matching panels':45s} {share:5.1f}%  [{low:.1f}, {high:.1f}]  ({len(pairs)} pairs)   paper: 79.6")

    half = len(pairs) // 2
    probe = fit_linear_probe(vectors[:half], holds_shape[:half])
    for pair, scores, truth in zip(pairs[half:], probe(vectors[half:]), holds_shape[half:]):
        for panel in range(6):                                  # one case per panel of a test pair
            results.add(pair["index"], probe_reads_panel=bool((scores[panel] > 0.5) == truth[panel]))
    results.report("probe_reads_panel", paper=97.7, label="probe: panel holds the shape, read right")
    results.save(attention_on_matching_panels={"percent": share, "low": low, "high": high})


if __name__ == "__main__":
    main()
