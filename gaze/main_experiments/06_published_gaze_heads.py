"""Our gaze heads against the published Gaze Heads (Section 6.1).

The published Gaze Heads (Gandikota and Bau, 2026) were found without patching, by where heads
look while the model describes comic panels. We give their top 100 the two patches of our gaze
heads at the colon token: the queries from the source (which should give K, the target's letter at
the source's panel), and the source's image values (which should give C, the source's letter at
the target's panel).

    python gaze/main_experiments/06_published_gaze_heads.py --num_pairs 500

Paper: the top 100 published heads give K in 64.2% of cases (our 32 gaze heads: 74.0%) and C in
50.0% (ours: 93.7%). 23 of our 32 gaze heads are among the top 100.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    published = HEADS["published_gaze"][:100]
    print(f"{len(set(HEADS['gaze']) & set(published))} of our {len(HEADS['gaze'])} gaze heads are in the published top 100 (paper: 23)")
    results = Results("published_gaze_heads", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, letters in both_directions(model, pair):
            patches = [make_patch(source, target, "query", "colon", heads=published),
                       make_patch(source, target, "image_values", "colon", heads=published)]
            queries, values = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], queries=queries == " " + letters["K"], image_values=values == " " + letters["C"],
                        answer_queries=queries, answer_values=values)
    results.report("queries", paper=64.2, label="published top 100, queries: pointer followed (K)")
    results.report("image_values", paper=50.0, label="published top 100, image values (C)")
    results.save()


if __name__ == "__main__":
    main()
