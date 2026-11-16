"""Five tests at the colon token for five head sets (Appendix F.1, Table 13).

This extends gaze/main_experiments/01 to 06. Those files give each test to our 32 gaze heads and
random heads. Here every test is given to five head sets: the 32 gaze heads, 32 random heads from
the same layers, the top 100 and top 200 published Gaze Heads, and the 8 heads of the blocking
search (01_blocking_search.py). The five tests, all at the colon token:
    queries      the set's queries come from the source             counts when the target answers K
    held         the binding heads move the pointer at the shape token, and the set's queries are
                 kept at the target's own values                    counts when the target keeps T
    values       the set reads the source's image values            counts when the target answers C
    blocked      the set cannot see the image                       counts when the target loses T
    only         only this set sees the image (as in main file 03)  counts when the target keeps T

    python gaze/appendix_experiments/02_head_sets_held_out.py --num_pairs 500

Paper (percent of the 1,000 held-out cases; queries / held / values / blocked / only):
    32 gaze heads              74.0  98.4  93.7  11.8  95.1
    random heads                0.0   5.9   0.0   0.2   4.8
    published top 100          64.2  95.2  50.0   1.4  97.4
    published top 200          64.9  96.2  88.6   1.1  98.6
    blocking-search heads       0.0   8.3   0.0   2.1   (not run)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import all_heads, load_model, n_layers
from utils.patching import Patch, make_patch, run_patches

PAPER = {"gaze": (74.0, 98.4, 93.7, 11.8, 95.1), "random": (0.0, 5.9, 0.0, 0.2, 4.8),
         "published_100": (64.2, 95.2, 50.0, 1.4, 97.4), "published_200": (64.9, 96.2, 88.6, 1.1, 98.6),
         "blocking_search": (0.0, 8.3, 0.0, 2.1, None)}
TESTS = ("queries", "held", "values", "blocked", "only")


def tests(model, source, target, heads, with_only=True):
    """The five patches for one head set, in the order of TESTS."""
    colon, image = target.token("colon"), target.image_positions
    held = make_patch(source, target, "head_output", "shape", heads=HEADS["binding"])          # move the pointer
    held.add(make_patch(target, target, "query", "colon", heads=heads))                        # keep these queries
    patches = [make_patch(source, target, "query", "colon", heads=heads), held,
               make_patch(source, target, "image_values", "colon", heads=heads), Patch().block(heads, [colon], image)]
    if with_only:
        only = Patch().block(None, range(target.token("shape") + 1, colon), image, layers=range(n_layers(model)))
        patches.append(only.block([h for h in all_heads(model) if h not in heads], [colon], image))
    return patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("head_sets_held_out", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        sets = {"gaze": HEADS["gaze"], "random": random_like(HEADS["gaze"], seed=pair["index"]),
                "published_100": HEADS["published_gaze_ranking"][:100], "published_200": HEADS["published_gaze_ranking"][:200],
                "blocking_search": HEADS["blocking_search"]}
        for source, target, letters in both_directions(model, pair):
            row = {}
            for name, heads in sets.items():
                answers = run_patches(model, target, tests(model, source, target, heads, with_only=name != "blocking_search"))["answers"]
                wanted = (letters["K"], letters["T"], letters["C"], letters["T"], letters["T"])
                for test, answer, letter in zip(TESTS, answers, wanted):
                    row[f"{name}_{test}"] = (answer != " " + letter) if test == "blocked" else (answer == " " + letter)
            results.add(pair["index"], **row)
    for name, paper in PAPER.items():
        for test, value in zip(TESTS, paper):
            if value is not None:
                results.report(f"{name}_{test}", paper=value)
    results.save()


if __name__ == "__main__":
    main()
