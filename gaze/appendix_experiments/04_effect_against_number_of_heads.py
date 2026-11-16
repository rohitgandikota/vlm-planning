"""Effect against the number of heads, for three rankings (Appendix F.2, Figure 4b).

This extends gaze/main_experiments/06_published_gaze_heads.py from 100 heads to every K. We take
the top K heads of three rankings of all 1,152 heads: our query-patch search
(HEADS["gaze_ranking"]), the published Gaze Heads ranking, and a random order of all heads, drawn
anew for every pair and direction. Each K gets two patches at the colon token, in separate runs:
    query    the heads' queries come from the source        counts when the target answers K
    values   the heads read the source's image values       counts when the target answers C

    python gaze/appendix_experiments/04_effect_against_number_of_heads.py --num_pairs 500

Paper: our 32 gaze heads give 74.0% (query) and 93.7% (values). The published top 100 give 64.2%
and 50.0%, the published top 256 give 63.8% and 89.1%. Random heads stay at or below 2.1% at every K.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import all_heads, load_model
from utils.patching import make_patch, run_patches

SIZES = (1, 2, 4, 8, 16, 32, 64, 100, 128, 200, 256)
PAPER = {("ours", 32): (74.0, 93.7), ("published", 100): (64.2, 50.0), ("published", 256): (63.8, 89.1)}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("effect_against_number_of_heads", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for direction, (source, target, letters) in enumerate(both_directions(model, pair)):
            shuffled = all_heads(model)
            random.Random(1000 * pair["index"] + direction).shuffle(shuffled)
            rankings = {"ours": HEADS["gaze_ranking"], "published": HEADS["published_gaze_ranking"], "random": shuffled}
            keys, patches = [], []
            for name, ranking in rankings.items():
                for k in SIZES:
                    keys += [f"{name}_{k}_query", f"{name}_{k}_values"]
                    patches += [make_patch(source, target, "query", "colon", heads=ranking[:k]),
                                make_patch(source, target, "image_values", "colon", heads=ranking[:k])]
            answers = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], **{key: answer == " " + letters["K" if key.endswith("query") else "C"]
                                          for key, answer in zip(keys, answers)})
    for name in ("ours", "published", "random"):
        for k in SIZES:
            paper = PAPER.get((name, k), (None, None))
            results.report(f"{name}_{k}_query", paper=paper[0], label=f"{name}, top {k}, queries (K)")
            results.report(f"{name}_{k}_values", paper=paper[1], label=f"{name}, top {k}, image values (C)")
    results.save()


if __name__ == "__main__":
    main()
