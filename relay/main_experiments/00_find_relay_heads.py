"""Find the relay heads (Sections 3 and 4.2).

The pointer is resolved at the shape token and used at the colon token, so some heads at the colon
token must fetch it. We search for them over all 1,152 heads. A head at the colon token keeps its
own query, keys and attention weights, but the values it reads from the shape token and the tokens
after it come from the source run. A head that carries the pointer makes the target answer K, its
own letter at the panel the source points to.

The search has three steps.
    1. On the search pairs, patch one head at a time and score it by how much the patch raises the
       logit of K over the target's own letter.
    2. On the validation pairs, patch the top K heads together for K = 1, 2, 4, ..., 128.
    3. Take the smallest K for which the target answers K in at least 80% of cases and the
       published Gaze Heads look at its panel in at least 80% of cases (the best K if none does).

    python relay/main_experiments/00_find_relay_heads.py

--num_pairs is the number of search pairs (the paper uses 200), and half as many validation pairs
are used. Step 1 is slow: 1,152 patched runs for each case. To run it in parts, give each part
--start and --num_pairs; a part only saves its scores, and a last run without --start reads them.

Paper: 16 relay heads, listed in utils/heads.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, search_heads, select_heads
from utils.model import load_model
from utils.patching import make_patch


def values_at_colon_token(source, target, heads):
    """At the colon token, the heads read the source's values from the shape token through the
    token before the colon token."""
    read_from = range(target.token("shape"), target.token("colon"))
    return make_patch(source, target, "values", "colon", heads=heads, positions=read_from)


def main():
    args = parse_args(__doc__, split="search", num_pairs=200)
    model = load_model(args.model, args.device)
    results = Results("find_relay_heads", args)
    search_pairs = one_answer_pairs(model, "search", args.start + args.num_pairs)[args.start:]
    scores = search_heads(model, search_pairs, {"relay": (values_at_colon_token, "K")},
                          folder=Path(args.output) / "find_relay_heads_scores")["relay"]
    if args.start:                                                  # one part of step 1: nothing to select yet
        return
    validation_pairs = one_answer_pairs(model, "validation", max(1, args.num_pairs // 2))
    heads, rates = select_heads(model, validation_pairs, scores, values_at_colon_token, "K", "relay")
    print(f"relay heads found: {heads}")
    print(f"same as the paper's {len(HEADS['relay'])} relay heads: {heads == HEADS['relay']}")
    results.save(found={"relay": {"heads": heads, "k_sweep": rates, "scores": scores.tolist()}})


if __name__ == "__main__":
    main()
