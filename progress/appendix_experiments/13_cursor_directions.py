"""Directions of the cursor in the comma token's residual (Appendix D.5, "Directions").

On 120 images with the asked shape at four panels, we write the model's own letters up to each
matching panel p and record the residual vector of the last comma token at layers 10 and 19 to 22
and 30. A run counts when the model then gives the letter of the next matching panel (a run done
up to panel 6 counts as it is). We also record the colon token with nothing written (p = 0).
    direction for p   the mean residual of the runs done up to p, minus the mean of these six means
    comma offset      the mean of the six means minus the mean colon-token residual
We print the rank of the six directions, their cosines, and the norms, and save every direction and
mean in directions.npz in the output folder. 14_writing_the_cursor.py and 15_cursor_subspace.py use it.

    python progress/appendix_experiments/13_cursor_directions.py --num_pairs 120

Paper: the six directions have rank 5 at every layer. Neighboring p are similar and distant p are
opposed (cosine 0.67 between panels 1 and 2, -0.73 between 1 and 5). The comma offset has norm 53 at
layer 20, larger than any direction.
The printed cosines came from an earlier, smaller run on images drawn from another seed.
--num_pairs is the number of images here, and --split is not used.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import cursor_images, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model

LAYERS = (10, 19, 20, 21, 22, 30)


def main():
    args = parse_args(__doc__, num_pairs=120)
    model = load_model(args.model, args.device)
    results = Results("cursor_directions", args)
    recorded = {layer: {p: [] for p in range(7)} for layer in LAYERS}      # p = 0 is the colon token
    for image in cursor_images("directions", args.num_pairs):
        letters, matching = image["letters"], image["matching"]
        for p in [0] + [panel + 1 for panel in matching]:                   # p counts panels from 1
            written = [letters[panel] for panel in matching if panel < p]
            after = [panel for panel in matching if panel >= p]
            run = run_list(model, image, written)
            counts = run.answer == " " + letters[after[0]] if after else p == 6
            results.add(image["index"], p=p, counts=counts)
            if counts:
                for layer in LAYERS:
                    recorded[layer][p].append(run.residual(layer, "last").float().cpu().numpy())
    saved = {}
    for layer in LAYERS:
        means = {p: np.mean(recorded[layer][p], axis=0) for p in range(1, 7) if recorded[layer][p]}
        grand = np.mean(list(means.values()), axis=0)
        directions = np.stack([means[p] - grand for p in means])
        singular = np.linalg.svd(directions, compute_uv=False)
        unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        cosines = unit @ unit.T
        offset = grand - np.mean(recorded[layer][0], axis=0)
        print(f"layer {layer}: rank {(singular > singular[0] * 1e-6).sum()} of {len(means)}, "
              f"cosine p1-p2 {cosines[0, 1]:.2f}, p1-p5 {cosines[0, 4]:.2f}, comma offset norm {np.linalg.norm(offset):.0f}, "
              f"direction norms {np.round(np.linalg.norm(directions, axis=1)).astype(int).tolist()}")
        saved[f"comma_mean_{layer}"], saved[f"colon_mean_{layer}"] = grand, np.mean(recorded[layer][0], axis=0)
        for p, direction in zip(means, directions):
            saved[f"direction_{p}_{layer}"] = direction
            # how far single runs lie from the mean comma state, relative to the direction (used by 14)
            saved[f"spread_{p}_{layer}"] = np.linalg.norm(np.stack(recorded[layer][p]) - grand, axis=1).mean() / np.linalg.norm(direction)
    print("paper: rank 5 at layers 19 to 22; cosine 0.67 (p1-p2), -0.73 (p1-p5); comma offset norm 53 at layer 20")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    np.savez(Path(args.output) / "directions.npz", **saved)
    results.save()


if __name__ == "__main__":
    main()
