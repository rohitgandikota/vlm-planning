"""The answer token records the written panel, and the comma token subtracts it (Appendix E.3, last paragraph).

No patch. With nothing written we take what the binding heads write into the residual stream at
the shape token (the sum of their outputs after the output projection). A ridge probe on the first
half of the pairs learns six panel directions, one per panel: "does this panel hold the shape we
ask for?". With W written we take what the same heads write at the answer token of W and at the
comma token after it, and score them in these directions on the second half of the pairs. Scores
are centered on the probe's mean. A panel that counts as matching scores high.

    python progress/appendix_experiments/50_panel_directions_at_answer_and_comma.py --num_pairs 500

Paper: at the answer token the written panel scores +0.50 and the next panel -0.09, and the next
panel scores below the written one in 85.6% of pairs. At the comma token the written panel scores
-0.33 and the next panel +0.20; the written panel scores below the panels without the shape in
66.0% of pairs and below the other matching panels in 89.2%. The binding heads here are the 16
binding heads with the query and key heads (18 heads, HEADS["binding_query_key"]), as in the run
that produced these numbers. Every held-out pair is used, the first 250 to fit and the last 250 to test.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import fit_linear_probe, written_by_heads


def written(model, run, token):
    """What the binding heads write at one token, stored in half precision as in the paper's run."""
    return written_by_heads(model, run, HEADS["binding_query_key"], token).half().cpu().numpy()


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("panel_directions_at_answer_and_comma", args)
    pairs = list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]
    at_shape, at_answer, at_comma = [], [], []
    for pair in pairs:
        nothing = run_list(model, pair["target"], [])
        one = run_list(model, pair["target"], [pair["written"]])
        at_shape.append(written(model, nothing, "shape"))
        at_answer.append(written(model, one, "last_answer"))
        at_comma.append(written(model, one, "last_comma"))
    matching = np.array([[panel in pair["panels"] for panel in range(6)] for pair in pairs], float)
    half = len(pairs) // 2
    probe = fit_linear_probe(np.array(at_shape[:half]), matching[:half])
    center = matching[:half].mean(0)
    scores = {"answer": probe(np.array(at_answer[half:])) - center, "comma": probe(np.array(at_comma[half:])) - center}
    for i, pair in enumerate(pairs[half:]):
        r = pair["panels"]
        fillers = [p for p in range(6) if p not in r]
        a, c = scores["answer"][i], scores["comma"][i]
        results.add(pair["index"], answer_next_below_written=bool(a[r[1]] < a[r[0]]),
                    comma_written_below_fillers=bool(c[r[0]] < c[fillers].mean()),
                    comma_written_below_other_matching=bool(c[r[0]] < c[r[2:]].mean()),
                    answer_written=float(a[r[0]]), answer_next=float(a[r[1]]),
                    comma_written=float(c[r[0]]), comma_next=float(c[r[1]]))
    rows = results.rows
    for name, paper in (("answer_written", "+0.50"), ("answer_next", "-0.09"), ("comma_written", "-0.33"), ("comma_next", "+0.20")):
        print(f"mean score, {name.replace('_', ' '):35s} {np.mean([row[name] for row in rows]):+.2f}   paper: {paper}")
    results.report("answer_next_below_written", paper=85.6, label="answer token: next below written")
    results.report("comma_written_below_fillers", paper=66.0, label="comma token: written below panels without the shape")
    results.report("comma_written_below_other_matching", paper=89.2, label="comma token: written below other matching")
    results.save()


if __name__ == "__main__":
    main()
