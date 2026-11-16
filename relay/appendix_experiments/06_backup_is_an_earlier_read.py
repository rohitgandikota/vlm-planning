"""The backup for the gaze heads is an earlier read (Appendix C.3, last paragraph).

This extends gaze/main_experiments/05_gaze_heads_blocked.py. The tokens between the shape token and
the colon token read the letter from the image, and the colon token can take it from them. No
source run is used. In the target's own run we block, from the image:
    colon_blocked     every head at the colon token
    between_blocked   every head at the tokens between the shape token and the colon token
    both_blocked      the 32 gaze heads at the colon token, and every head at the tokens in between
and count the cases that keep the target's own answer T. If either route alone is enough, only
the last condition breaks the answer.

    python relay/appendix_experiments/06_backup_is_an_earlier_read.py --num_pairs 500

Paper: T is kept in 98.5% (colon token blocked), 96.4% (tokens in between blocked) and 2.6% (both).
These numbers come from an earlier, smaller run (196 cases of 100 pairs of an earlier pair set)
with an earlier gaze-head list that shares 31 of its 32 heads with HEADS["gaze"].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model, n_layers
from utils.patching import Patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    layers = range(n_layers(model))
    results = Results("backup_is_an_earlier_read", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for _, target, letters in both_directions(model, pair):     # each side of the pair is a target once
            shape, colon, image = target.token("shape"), target.token("colon"), target.image_positions
            between = range(shape + 1, colon)
            patches = {"colon_blocked": Patch().block(None, [colon], image, layers=layers),
                       "between_blocked": Patch().block(None, between, image, layers=layers),
                       "both_blocked": Patch().block(HEADS["gaze"], [colon], image).block(None, between, image, layers=layers)}
            answers = run_patches(model, target, list(patches.values()))["answers"]
            results.add(pair["index"], **{name: answer == " " + letters["T"] for name, answer in zip(patches, answers)})
    results.report("colon_blocked", paper=98.5, label="colon token blocked: own answer kept (T)")
    results.report("between_blocked", paper=96.4, label="tokens in between blocked: own answer kept (T)")
    results.report("both_blocked", paper=2.6, label="gaze heads and tokens in between blocked (T)")
    results.save()


if __name__ == "__main__":
    main()
