"""Find the binding heads, the query heads and the key heads (Sections 3 and 4.1).

Three searches at the shape token, each over all 1,152 heads:
    binding   the output of a head is patched from the source. A head that holds the pointer makes
              the target answer K, its own letter at the panel the source points to.
    query     the query of a head is patched. A head whose query carries the request makes the
              target answer P, its letter inside the shape the source asks for.
    key       a head reads the source's image keys. A head that reads the layout through its keys
              makes the target answer R, its letter at the panel where the source holds its shape.

Each search has three steps.
    1. On the search pairs, patch one head at a time and score it by how much the patch raises the
       logit of the expected letter over the target's own letter.
    2. On the validation pairs, patch the top K heads together for K = 1, 2, 4, ..., 128.
    3. Take the smallest K for which the expected letter comes out in at least 80% of cases and the
       published Gaze Heads look at its panel in at least 80% of cases (the best K if none does).

    python binding/main_experiments/00_find_binding_heads.py

--num_pairs is the number of search pairs (the paper uses 200), and half as many validation pairs
are used. Step 1 is slow: 3 x 1,152 patched runs for each case. To run it in parts, give each part
--start and --num_pairs; a part only saves its scores, and a last run without --start reads them.

Paper: 16 binding heads, 8 query heads and 8 key heads, listed in utils/heads.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, search_heads, select_heads
from utils.model import load_model
from utils.patching import make_patch


def patch_at_shape_token(component):
    """A function that patches one component of the given heads at the shape token."""
    return lambda source, target, heads: make_patch(source, target, component, "shape", heads=heads)


# name of the head set: (how its heads are patched, the outcome letter the patch should give)
SEARCHES = {"binding": (patch_at_shape_token("head_output"), "K"),
            "query": (patch_at_shape_token("query"), "P"),
            "key": (patch_at_shape_token("image_keys"), "R")}


def main():
    args = parse_args(__doc__, split="search", num_pairs=200)
    model = load_model(args.model, args.device)
    results = Results("find_binding_heads", args)
    search_pairs = one_answer_pairs(model, "search", args.start + args.num_pairs)[args.start:]
    scores = search_heads(model, search_pairs, SEARCHES, folder=Path(args.output) / "find_binding_heads_scores")
    if args.start:                                                  # one part of step 1: nothing to select yet
        return
    validation_pairs = one_answer_pairs(model, "validation", max(1, args.num_pairs // 2))
    found = {}
    for name, (build, letter) in SEARCHES.items():
        heads, rates = select_heads(model, validation_pairs, scores[name], build, letter, name)
        found[name] = {"heads": heads, "k_sweep": rates, "scores": scores[name].tolist()}
        print(f"{name} heads found: {heads}")
        print(f"same as the paper's {len(HEADS[name])} {name} heads: {heads == HEADS[name]}")
    results.save(found=found)


if __name__ == "__main__":
    main()
