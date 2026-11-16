"""Blocking the gaze heads from the image: sufficient but not necessary (Section 6.1).

We block the attention of the 32 gaze heads to the image at the colon token and ask whether the
target still answers correctly (outcome T). Most answers survive, because the tokens between the
shape token and the colon token also read the letter from the image, and the colon token can take
it from them. So in a second run we block these tokens from the image as well (every head).

    python gaze/main_experiments/05_gaze_heads_blocked.py --num_pairs 500

Paper: 88.2% of cases still give T with the gaze heads blocked, and 2.6% when the tokens between
the shape token and the colon token are blocked as well. The 2.6% in the paper comes from a
smaller, earlier set (196 cases of 100 pairs) and an earlier gaze-head list that differs in one
head. This file runs the same test on the standard held-out pairs, so expect a value near 2.6%,
not exactly 2.6%.
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
    results = Results("gaze_heads_blocked", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, letters in both_directions(model, pair):
            shape, colon, image = target.token("shape"), target.token("colon"), target.image_positions
            gaze_blocked = Patch().block(HEADS["gaze"], [colon], image)
            both_blocked = Patch().block(HEADS["gaze"], [colon], image)
            both_blocked.block(None, range(shape + 1, colon), image, layers=range(n_layers(model)))
            first, second = run_patches(model, target, [gaze_blocked, both_blocked])["answers"]
            results.add(pair["index"], gaze_heads_blocked=first == " " + letters["T"],
                        tokens_between_blocked_too=second == " " + letters["T"], answer=first)
    results.report("gaze_heads_blocked", paper=88.2, label="gaze heads blocked: still correct (T)")
    results.report("tokens_between_blocked_too", paper="2.6 (earlier, smaller set)", label="tokens between blocked too: still correct (T)")
    results.save()


if __name__ == "__main__":
    main()
