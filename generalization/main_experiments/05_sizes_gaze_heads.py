"""The gaze heads read the pointer at every model size (Appendix G.1.3, Table "sizes-gaze", rows 1 to 3).

With each size's own gaze heads (found by the query patch at the colon token), as in
gaze/main_experiments/01, 02 and 04:
    queries patched     the gaze heads' queries at the colon token come from the source: the target
                        answers K, its letter at the panel the source points to
    queries held        the binding heads' outputs move the pointer at the shape token, but the gaze
                        heads keep the target's own queries: the target keeps its answer T
    image values        the gaze heads read the source's image values at the colon token: the target
                        answers C, the source's letter at the target's panel
The control patches the queries of random heads from the same layers (seed = pair index). We also
patch the queries of the published 8B Gaze Heads applied to this size by head index (the top 100,
keeping the heads that exist on this size).

    python generalization/main_experiments/05_sizes_gaze_heads.py --model Qwen/Qwen3-VL-4B-Instruct --num_pairs 500

Models: any Qwen3-VL size. 2B 16 GB, 4B 24 GB, 8B 48 GB, 32B 160 GB (one B200) or --device cpu.
5 patched runs per case, 2 cases per pair.

Paper (2B with 16 heads / 4B with 128 / 8B with 32 / 32B with 64; the 2B and 4B on 100 earlier
held-out pairs, the 32B on 20): queries patched 63.5 / 75.5 / 74.0 / 90.0; queries held 82.0 / 92.0 /
98.4 / 100.0; image values 44.5 / 63.0 / 93.7 / 97.5; random heads 0.0 at every size. The published
8B ranking by index: 10.0 (2B), 56.0 (4B).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, pairs_of_8b
from utils.experiment import Results, parse_args
from utils.heads import HEADS, heads_of_size, random_heads_like
from utils.model import load_model, n_heads, n_layers
from utils.patching import make_patch, run_patches

PAPER = {"queries_patched": {"2B": 63.5, "4B": 75.5, "8B": 74.0, "32B": 90.0},
         "queries_held": {"2B": 82.0, "4B": 92.0, "8B": 98.4, "32B": 100.0},
         "image_values": {"2B": 44.5, "4B": 63.0, "8B": 93.7, "32B": 97.5},
         "published_8b_heads": {"2B": 10.0, "4B": 56.0}}


def main():
    args = parse_args(__doc__)
    pairs = pairs_of_8b("one_answer", args.split, args.start + args.num_pairs, args.device)[args.start:]
    model = load_model(args.model, args.device)
    gaze, binding = heads_of_size(args.model, "gaze"), heads_of_size(args.model, "binding")
    published = [(l, h) for l, h in HEADS["published_gaze"] if l < n_layers(model) and h < n_heads(model)]
    results = Results(f"sizes_gaze_heads_{args.model.split('/')[-1]}", args)
    for pair in pairs:
        random_heads = random_heads_like(model, gaze, seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            held = make_patch(source, target, "head_output", "shape", heads=binding)        # move the pointer
            held.add(make_patch(target, target, "query", "colon", heads=gaze))              # keep the gaze queries
            patches = {"queries_patched": (make_patch(source, target, "query", "colon", heads=gaze), "K"),
                       "random_heads": (make_patch(source, target, "query", "colon", heads=random_heads), "K"),
                       "published_8b_heads": (make_patch(source, target, "query", "colon", heads=published), "K"),
                       "queries_held": (held, "T"),
                       "image_values": (make_patch(source, target, "image_values", "colon", heads=gaze), "C")}
            answers = run_patches(model, target, [patch for patch, _ in patches.values()])["answers"]
            results.add(pair["index"], **{name: answer == " " + letters[letter]
                                          for (name, (_, letter)), answer in zip(patches.items(), answers)})
    size = args.model.split("-")[-2]
    results.report("queries_patched", paper=PAPER["queries_patched"].get(size), label="gaze queries patched: K")
    results.report("random_heads", paper=0.0, label="random heads' queries patched: K")
    results.report("published_8b_heads", paper=PAPER["published_8b_heads"].get(size), label=f"published 8B heads ({len(published)}): K")
    results.report("queries_held", paper=PAPER["queries_held"].get(size), label="pointer moved, gaze queries held: T")
    results.report("image_values", paper=PAPER["image_values"].get(size), label="gaze heads read the source's image values: C")
    results.save(gaze_heads=gaze)


if __name__ == "__main__":
    main()
