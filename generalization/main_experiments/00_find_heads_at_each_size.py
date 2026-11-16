"""Find the head sets of one Qwen3-VL size (Appendix G.1).

We carry no head over between model sizes. On each size we run the searches of the main paper
again, on one-answer pairs, over every head of that size:
    binding   head outputs at the shape token, outcome K (the pointer moved)
    query     queries at the shape token, outcome P (the request moved)
    key       image keys read at the shape token, outcome R (the layout moved)
    relay     values read at the colon token from the shape token on, outcome K
    gaze      queries at the colon token, outcome K
Step 1 scores one head at a time on the search pairs (gain in the logit of the outcome letter over
the target's own letter). Step 2 patches the top K heads together on the validation pairs. Step 3
takes the smallest K whose outcome rate reaches 80%, or the best K if none does. The 8B search also
asks that the published Gaze Heads look at the right panel; those heads exist only on the 8B, so
here K is chosen on the outcome rate alone, as in the paper's runs on the other sizes.

    python generalization/main_experiments/00_find_heads_at_each_size.py --model Qwen/Qwen3-VL-2B-Instruct --num_pairs 100

--num_pairs is the number of search pairs; half as many validation pairs are used. The paper used
100 and 50 on the 2B and 4B, and 20 and 10 on the 32B. The pairs are the 8B's pairs (see
`pairs_of_8b`). To run step 1 in parts, give each part --start and --num_pairs; a part only saves
its scores, and a last run without --start reads them.
Models: Qwen3-VL-2B (one GPU, 16 GB), 4B (24 GB), 32B (one GPU with 160 GB, such as a B200, or the
CPU with --device cpu). Step 1 takes 5 x (number of heads) patched runs per case: 4,480 on the 2B,
11,520 on the 4B, 40,960 on the 32B.

Paper: the sets listed in utils/heads.py. Gaze heads: 16 on the 2B (no K passes),
128 on the 4B (no K passes), 64 on the 32B (95% on the validation pairs). On the 4B the paper's
relay set has 16 heads; the rule on the outcome rate alone gives 8.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import pairs_of_8b
from utils.experiment import Results, parse_args
from utils.heads import K_VALUES, choose_k, rank_heads, search_heads, sweep_k_letter
from utils.model import load_model, n_heads, n_layers
from utils.patching import make_patch


def patch_at(component, token):
    """A function that patches one component of the given heads at one token."""
    return lambda source, target, heads: make_patch(source, target, component, token, heads=heads)


def relay_values(source, target, heads):
    """At the colon token, the heads read the source's values from the shape token on."""
    return make_patch(source, target, "values", "colon", heads=heads,
                      positions=range(target.token("shape"), target.token("colon")))


SEARCHES = {"binding": (patch_at("head_output", "shape"), "K"),
            "query": (patch_at("query", "shape"), "P"),
            "key": (patch_at("image_keys", "shape"), "R"),
            "relay": (relay_values, "K"),
            "gaze": (patch_at("query", "colon"), "K")}


def main():
    args = parse_args(__doc__, split="search", num_pairs=100)
    search_pairs = pairs_of_8b("one_answer", "search", args.start + args.num_pairs, args.device)[args.start:]
    validation_pairs = pairs_of_8b("one_answer", "validation", max(1, args.num_pairs // 2), args.device)
    model = load_model(args.model, args.device)
    results = Results(f"find_heads_{args.model.split('/')[-1]}", args)
    scores = search_heads(model, search_pairs, SEARCHES, folder=Path(args.output) / f"find_heads_{args.model.split('/')[-1]}")
    if args.start:                                                  # one part of step 1: nothing to select yet
        return
    found = {}
    for name, (build, letter) in SEARCHES.items():
        ranking = rank_heads(scores[name])
        # K = 1, 2, 4, ..., 128 (256 for the gaze heads), at most half of the model's heads
        sizes = [k for k in (K_VALUES + (256,) if name == "gaze" else K_VALUES) if k <= n_layers(model) * n_heads(model) // 2]
        rates = sweep_k_letter(model, validation_pairs, ranking, build, letter, sizes)
        k = choose_k(rates)
        print(f"\n{name}: rate of {letter} by K on {len(validation_pairs)} validation pairs")
        for size, rate in rates.items():
            print(f"  K = {size:3d}   {100 * rate:5.1f}%" + ("   <- chosen" if size == k else ""))
        print(f"{name} heads found: {ranking[:k]}")
        found[name] = {"heads": ranking[:k], "k_sweep": rates, "ranking": ranking[:256]}
    results.save(found=found)


if __name__ == "__main__":
    main()
