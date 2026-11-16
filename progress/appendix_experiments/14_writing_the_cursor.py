"""Writing the cursor with no source (Appendix D.5, "Writing the cursor").

On 500 fresh images with nothing written, we add a vector to the residual vector of the colon
token at layer 21 and read the next letter. The vectors come from 13_cursor_directions.py (run it
first with the same --output folder): the direction for panel p, and the comma offset (the mean
comma state minus the mean colon state). An image counts for p when its first matching panel is
not after p and a matching panel lies after p. A case is a success when the target starts at the
first matching panel after p.
    direction             the direction for p alone
    random                a random direction of the same norm (five draws)
    offset                the comma offset alone
    offset_direction      the comma offset plus the direction for p
    offset_random         the comma offset plus a random direction of the same spread (five draws)
    comma_state           the colon residual replaced by the mean comma state for p
    from_layer_10, 30     the direction for p taken at layer 10 or 30, added at layer 20

    python progress/appendix_experiments/14_writing_the_cursor.py --num_pairs 500

Paper: the direction alone gives 0.0 to 2.2% successes, the same as a random direction. The offset
alone sends every target near panel 4 (mean chosen panel 4.2 to 4.6 for every p). The offset plus
the direction gives 97.7%, 96.3%, 96.8%, 96.1%, and 96.5% for p = 1 to 5, with the mean chosen panel
at 2.49, 3.42, 4.38, 5.26, and 5.95. The offset plus a random direction gives 8.8%, 38.0%, 63.3%,
48.3%, and 15.7% (mean of five draws). The mean comma state gives the same numbers as the offset plus
the direction. The direction from layer 10 does nothing, and the one from layer 30 overshoots.
--num_pairs is the number of images here, and --split is not used.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import cursor_images, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model
from utils.patching import Patch, run_patches

LAYER = 21
PAPER = {1: (97.7, 2.49, 8.8), 2: (96.3, 3.42, 38.0), 3: (96.8, 4.38, 63.3), 4: (96.1, 5.26, 48.3), 5: (96.5, 5.95, 15.7)}


def random_direction(size, norm, seed):
    """A random direction with a given norm, the same for the same seed."""
    vector = torch.randn(size, generator=torch.Generator().manual_seed(seed))
    return vector / vector.norm() * norm


def vectors(saved, p):
    """(name, layer, vector to add) for every condition of panel p. Seeds follow the paper's run."""
    d = lambda layer: torch.from_numpy(saved[f"direction_{p}_{layer}"])
    direction, spread = d(LAYER), float(saved[f"spread_{p}_{LAYER}"])
    offset = torch.from_numpy(saved[f"comma_mean_{LAYER}"] - saved[f"colon_mean_{LAYER}"])
    out = [("direction", LAYER, direction), ("offset", LAYER, offset), ("offset_direction", LAYER, offset + direction),
           ("from_layer_10", 20, d(10)), ("from_layer_30", 20, d(30))]
    for draw in range(5):
        noise = random_direction(direction.numel(), float(direction.norm()), 1000 * p + 10 * LAYER + draw)
        out += [(f"random_{draw}", LAYER, noise), (f"offset_random_{draw}", LAYER, offset + spread * noise)]
    return out


def main():
    args = parse_args(__doc__)
    saved_file = Path(args.output) / "directions.npz"
    if not saved_file.exists():
        sys.exit(f"{saved_file} not found. Run progress/appendix_experiments/13_cursor_directions.py first.")
    saved = np.load(saved_file)
    model = load_model(args.model, args.device)
    results = Results("writing_the_cursor", args)
    for image in cursor_images("test", args.num_pairs):
        letters, matching = image["letters"], image["matching"]
        target = run_list(model, image, [])
        if target.answer != " " + letters[matching[0]]:
            continue
        colon, row = target.token("colon"), {}
        for p in range(1, 6):
            if matching[0] > p - 1 or not any(panel > p - 1 for panel in matching):
                continue
            first_after = letters[next(panel for panel in matching if panel > p - 1)]
            conditions = vectors(saved, p)
            patches = [Patch().residual(layer, colon, target.residual(layer, colon) + vector.to(target.device))
                       for _, layer, vector in conditions]
            comma_state = torch.from_numpy(saved[f"comma_mean_{LAYER}"] + saved[f"direction_{p}_{LAYER}"])
            patches.append(Patch().residual(LAYER, colon, comma_state.to(target.device)))
            names = [name for name, _, _ in conditions] + ["comma_state"]
            for name, answer in zip(names, run_patches(model, target, patches)["answers"]):
                row[f"p{p}_{name}"] = answer == " " + first_after
                row[f"p{p}_{name}_panel"] = letters.index(answer.strip()) + 1 if answer.strip() in letters else 0
        results.add(image["index"], **row)
    for p, (success, panel, random_success) in PAPER.items():
        mean_panel = lambda name: np.mean([r[f"p{p}_{name}_panel"] for r in results.rows if r.get(f"p{p}_{name}_panel")] or [0])
        results.report(f"p{p}_direction", paper="0.0 to 2.2", label=f"p = {p}: direction alone")
        results.report(f"p{p}_offset_direction", paper=success, label=f"p = {p}: offset plus direction")
        results.report(f"p{p}_comma_state", paper=success, label=f"p = {p}: mean comma state")
        random = np.mean([results.rate(f"p{p}_offset_random_{draw}")["percent"] or 0 for draw in range(5)])
        print(f"{f'p = {p}: offset plus random (mean of 5)':45s} {random:5.1f}%   paper: {random_success}")
        print(f"{f'p = {p}: mean chosen panel':45s} offset alone {mean_panel('offset'):.2f}, offset plus direction "
              f"{mean_panel('offset_direction'):.2f}   paper: 4.2 to 4.6, {panel}")
        for name in ("random_0", "from_layer_10", "from_layer_30"):
            results.report(f"p{p}_{name}", label=f"p = {p}: {name}")
    results.save()


if __name__ == "__main__":
    main()
