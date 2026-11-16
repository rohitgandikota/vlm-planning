"""The binding patches at every model size (Appendix G.1.2, Table "sizes-one", first rows).

The four patches of Section 4.1 at the shape token, each with the model's own head set:
    queries of the query heads           the request moves: outcome P
    image keys read by the key heads     the layout moves: outcome R
    both, on the query and key heads     the pointer moves: outcome K
    outputs of the binding heads         the pointer moves: outcome K
and each one again with random heads from the same layers (seed = pair index). Every pair counts,
also when this size gets its own answer wrong with no patch.

    python generalization/main_experiments/02_sizes_binding_heads.py --model Qwen/Qwen3-VL-2B-Instruct --num_pairs 500

Models: any Qwen3-VL size. 2B 16 GB, 4B 24 GB, 8B 48 GB, 32B 160 GB (one B200) or --device cpu.
8 patched runs per case, 2 cases per pair.

Paper (2B / 4B / 8B / 32B; the 2B and 4B on 100 earlier held-out pairs, the 32B on 50):
queries 37.5 / 59.0 / 81.0 / 82.0; image keys 71.0 / 82.0 / 83.1 / 94.0; queries and keys
84.0 / 96.0 / 95.0 / 99.0; outputs 86.0 / 93.5 / 85.9 / 87.0; random heads at most 1.0 / 1.5 / 0.1 / 0.0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, pairs_of_8b
from utils.experiment import Results, parse_args
from utils.heads import heads_of_size, random_heads_like
from utils.model import load_model
from utils.patching import make_patch, run_patches

PAPER = {"query": {"2B": 37.5, "4B": 59.0, "8B": 81.0, "32B": 82.0}, "key": {"2B": 71.0, "4B": 82.0, "8B": 83.1, "32B": 94.0},
         "query_and_key": {"2B": 84.0, "4B": 96.0, "8B": 95.0, "32B": 99.0}, "binding": {"2B": 86.0, "4B": 93.5, "8B": 85.9, "32B": 87.0}}


def build(source, target, name, heads):
    """The patch of one row at the shape token."""
    if name == "query":
        return make_patch(source, target, "query", "shape", heads=heads)
    if name == "key":
        return make_patch(source, target, "image_keys", "shape", heads=heads)
    if name == "query_and_key":
        return make_patch(source, target, "query", "shape", heads=heads).add(make_patch(source, target, "image_keys", "shape", heads=heads))
    return make_patch(source, target, "head_output", "shape", heads=heads)


EXPECTED = {"query": "P", "key": "R", "query_and_key": "K", "binding": "K"}


def main():
    args = parse_args(__doc__)
    pairs = pairs_of_8b("one_answer", args.split, args.start + args.num_pairs, args.device)[args.start:]
    model = load_model(args.model, args.device)
    sets = {name: heads_of_size(args.model, name) for name in EXPECTED}
    results = Results(f"sizes_binding_heads_{args.model.split('/')[-1]}", args)
    for pair in pairs:
        random_sets = {name: random_heads_like(model, heads, seed=pair["index"]) for name, heads in sets.items()}
        for source, target, letters in both_directions(model, pair):
            patches = [build(source, target, name, sets[name]) for name in EXPECTED]
            patches += [build(source, target, name, random_sets[name]) for name in EXPECTED]
            answers = run_patches(model, target, patches)["answers"]
            row = {name: answer == " " + letters[EXPECTED[name]] for name, answer in zip(EXPECTED, answers[:4])}
            row.update({f"random_{name}": answer == " " + letters[EXPECTED[name]] for name, answer in zip(EXPECTED, answers[4:])})
            results.add(pair["index"], **row, answers=answers)
    size = args.model.split("-")[-2]
    for name, letter in EXPECTED.items():
        results.report(name, paper=PAPER[name].get(size), label=f"{name} heads patched: outcome {letter}")
        results.report(f"random_{name}", label=f"random heads for {name}: outcome {letter}")
    results.save(heads=sets)


if __name__ == "__main__":
    main()
