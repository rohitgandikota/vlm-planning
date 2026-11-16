"""Heads ranked by the Gaze score on our own task, against the gaze heads (Appendix F.2, Table 14).

The comparison with the published Gaze Heads changes two things at once: how the heads were chosen
(by patching or by attention) and on which task (ours or comic strips). Here we remove the second
difference. We take the top 32 and the top 100 heads of the Gaze score measured on our task
(HEADS["gaze_score_on_this_task"], see 03_gaze_score_on_this_task.py) and give them the two patches
of the gaze heads at the colon token:
    queries   the heads' queries come from the source        counts when the target answers K
    values    the heads read the source's image values       counts when the target answers C
Next to them: our 32 gaze heads, the published top 100, and random heads from the same layers as
the re-measured top 32 and top 100.

    python gaze/appendix_experiments/05_same_ranking_on_same_task.py --num_pairs 500

Paper (queries / values): re-measured top 32 63.3% / 64.8%, re-measured top 100 65.3% / 93.4%,
against 74.0% / 93.7% for the 32 gaze heads and 64.2% / 50.0% for the published top 100; 32 random
heads 0.0% / 0.0%. The re-measured rows come from an earlier, smaller run: 196 cases of 100 pairs of
an earlier pair set, with an earlier list of gaze heads that shares 31 of its 32 heads with
HEADS["gaze"]. Expect values near, not equal to, the paper's.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches

PAPER = {"measured_32": (63.3, 64.8), "measured_100": (65.3, 93.4), "gaze": (74.0, 93.7),
         "published_100": (64.2, 50.0), "random_32": (0.0, 0.0), "random_100": (None, None)}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    measured = HEADS["gaze_score_on_this_task"]
    results = Results("same_ranking_on_same_task", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        sets = {"measured_32": measured[:32], "measured_100": measured[:100], "gaze": HEADS["gaze"],
                "published_100": HEADS["published_gaze"][:100],
                "random_32": random_like(measured[:32], seed=pair["index"]),
                "random_100": random_like(measured[:100], seed=pair["index"])}
        for source, target, letters in both_directions(model, pair):
            patches = []
            for heads in sets.values():
                patches += [make_patch(source, target, "query", "colon", heads=heads),
                            make_patch(source, target, "image_values", "colon", heads=heads)]
            answers = run_patches(model, target, patches)["answers"]
            row = {}
            for i, name in enumerate(sets):
                row[f"{name}_queries"] = answers[2 * i] == " " + letters["K"]
                row[f"{name}_values"] = answers[2 * i + 1] == " " + letters["C"]
            results.add(pair["index"], **row)
    for name, (queries, values) in PAPER.items():
        results.report(f"{name}_queries", paper=queries)
        results.report(f"{name}_values", paper=values)
    results.save()


if __name__ == "__main__":
    main()
