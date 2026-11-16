"""Binding heads write where the answer is, not what is there (Section 4.1).

At the shape token, we let the 16 binding heads read the source's image values instead of the
target's. The heads keep their own queries, keys and attention weights, so they still attend to
the target's panel. If their output held the content of that panel, the answer would change. If it
holds only the location, which is the same in both images, the target keeps its own answer
(outcome T).

    python binding/main_experiments/07_binding_heads_image_values.py --num_pairs 500

Paper: the answer is unchanged in 95.4% of cases. The same patch on the 10 query and key heads
leaves it unchanged in 96.8% (appendix), and on random heads in 100.0%. As in the paper's run, the
random heads are drawn to match the 10 query and key heads.
Appendix B.4 adds one patch: the 10 query and key heads take the source's queries, image keys and
image values together, which moves the pointer (K) in 82.6% (random heads: at most 0.1%).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("binding_heads_image_values", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["query_and_key"], seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            patches = [make_patch(source, target, "image_values", "shape", heads=heads)
                       for heads in (HEADS["binding"], HEADS["query_and_key"], random_heads)]
            # Appendix B.4: queries, image keys and image values of the same heads, all from the source
            patches += [make_patch(source, target, "image_values", "shape", heads=heads)
                        .add(make_patch(source, target, "query", "shape", heads=heads))
                        .add(make_patch(source, target, "image_keys", "shape", heads=heads))
                        for heads in (HEADS["query_and_key"], random_heads)]
            binding, query_key, control, all_three, all_three_random = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], binding_heads=binding == " " + letters["T"],
                        query_and_key_heads=query_key == " " + letters["T"],
                        random_heads=control == " " + letters["T"], answer=binding,
                        with_queries_and_keys=all_three == " " + letters["K"],
                        with_queries_and_keys_random=all_three_random == " " + letters["K"])
    results.report("binding_heads", paper=95.4, label="binding heads: answer unchanged (T)")
    results.report("query_and_key_heads", paper=96.8, label="query and key heads: answer unchanged (T)")
    results.report("random_heads", paper=100.0, label="random heads: answer unchanged (T)")
    results.report("with_queries_and_keys", paper=82.6, label="queries, keys and values: pointer moved (K)")
    results.report("with_queries_and_keys_random", paper="at most 0.1", label="random heads: pointer moved (K)")
    results.save()


if __name__ == "__main__":
    main()
