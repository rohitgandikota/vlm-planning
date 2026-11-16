"""The cursor takes five dimensions (Appendix D.5, "Five dimensions").

Let S be the span of the six cursor directions of 13_cursor_directions.py (run it first with the
same --output folder). We use the pairs of Section 5.2: the source has its shape at panels 2, 4 and
5, the target at panels 1, 2, 3 and 6, with nothing written. In design p4 the source has written
panels 2 and 4, so its cursor sends the target to panel 6. In design p2 it has written panel 2
only, so its cursor sends the target to panel 3, while the mean comma state sends it to panel 6.
At layer 21 we change the target's colon-token residual by part of the difference to the source's
comma-token residual:
    full                  the whole difference (the patch of Section 5.2)
    span                  only its component inside S
    complement            only its component outside S
    span_mean_rest        the source's part in S, with the rest set to the mean comma state's
    complement_mean_span  the source's part outside S, with the part in S set to the mean's
    random                the component inside a random subspace of the same size (five draws)

    python progress/appendix_experiments/15_cursor_subspace.py --num_pairs 500

Paper: the part in S alone changes nothing (the own answer in 100.0%). With the rest from the mean,
the source's part in S gives the source's cursor in 68.8% (design p2). The source's complement with
the mean's part in S gives the mean's answer, panel 6, in 96.0% (design p2). The full patch gives the
source's cursor in 87.4% (p2) to 96.6% (p4). A random five-dimensional subspace moves nothing (0.0%).
These pairs are drawn for this experiment (utils.dataset.subspace_pairs), so --split is not used.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import name_answer, run_list, subspace_pairs
from utils.experiment import Results, parse_args
from utils.model import load_model
from utils.patching import Patch, run_patches

LAYER = 21


def span_basis(saved):
    """An orthonormal basis of the span of the six directions at LAYER, one row per dimension."""
    directions = np.stack([saved[f"direction_{p}_{LAYER}"] for p in range(1, 7) if f"direction_{p}_{LAYER}" in saved])
    _, singular, rows = np.linalg.svd(directions, full_matrices=False)
    return torch.from_numpy(rows[: int((singular > singular[0] * 1e-6).sum())].copy())


def random_basis(size, dimensions, seed):
    q, _ = torch.linalg.qr(torch.randn(size, dimensions, generator=torch.Generator().manual_seed(seed)))
    return q.T


def changes(source, target, basis, comma_mean):
    """The vector added to the target's colon-token residual in each condition."""
    inside = lambda vector, b=basis: b.T @ (b @ vector)
    own = target.residual(LAYER, "colon").float().cpu()
    delta = source.residual(LAYER, "last_comma").float().cpu() - own
    to_mean = comma_mean - own
    out = {"full": delta, "span": inside(delta), "complement": delta - inside(delta),
           "span_mean_rest": inside(delta) + to_mean - inside(to_mean),
           "complement_mean_span": delta - inside(delta) + inside(to_mean)}
    for draw in range(5):
        out[f"random_{draw}"] = inside(delta, random_basis(delta.numel(), len(basis), 900 + 10 * LAYER + draw))
    return out


def main():
    args = parse_args(__doc__)
    saved_file = Path(args.output) / "directions.npz"
    if not saved_file.exists():
        sys.exit(f"{saved_file} not found. Run progress/appendix_experiments/13_cursor_directions.py first.")
    saved = np.load(saved_file)
    basis, comma_mean = span_basis(saved), torch.from_numpy(saved[f"comma_mean_{LAYER}"])
    print(f"S has {len(basis)} dimensions at layer {LAYER}   paper: 5")
    model = load_model(args.model, args.device)
    results = Results("cursor_subspace", args)
    for design in ("p4", "p2"):
        for pair in subspace_pairs(model, design, args.start + args.num_pairs)[args.start:]:
            source, target = run_list(model, pair["source"]), run_list(model, pair["target"])
            vectors = changes(source, target, basis, comma_mean)
            patches = [Patch().residual(LAYER, target.token("colon"), target.residual(LAYER, "colon") + v.to(target.device))
                       for v in vectors.values()]
            said = dict(zip(vectors, [name_answer(a, pair["letters"]) for a in run_patches(model, target, patches)["answers"]]))
            results.add(f"{design}_{pair['index']}", design=design, said=said,
                        **{f"{design}_{name}_{outcome}": said[name] == outcome for name in vectors for outcome in pair["letters"]})
    results.report("p4_span_set", paper=100.0, label="p4: part in S alone, own answer")
    results.report("p2_span_mean_rest_position", paper=68.8, label="p2: part in S, rest from the mean: source's cursor")
    results.report("p2_complement_mean_span_panel_6", paper=96.0, label="p2: complement, part in S from the mean: panel 6")
    results.report("p2_full_position", paper=87.4, label="p2: full patch, source's cursor")
    results.report("p4_full_position", paper=96.6, label="p4: full patch, source's cursor")
    for draw in range(5):
        results.report(f"p4_random_{draw}_set", paper=100.0, label=f"p4: random subspace {draw}, own answer kept")
    results.save()


if __name__ == "__main__":
    main()
