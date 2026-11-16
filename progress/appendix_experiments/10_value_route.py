"""The first route carries the cursor, and it is spread over many heads (Appendix D.4, second paragraph).

The target has written W, its letter at r1, so it answers next with its letter at r2. The source
is the standard source: it holds W at r2 and its shape at r2 and r4, so its cursor is at r2. The
comma token can learn where W came from by reading the answer token (route 1) or by finding W in
the image itself (route 2). The target's next letter is named by what moved:
    own      its letter at r2 (nothing moved)
    cursor   its letter at r3 (the source's cursor moved in)
    next     its letter at r4 (the source's own next panel moved in)
The patches:
    route 1    every head at the comma token reads the source's values of the answer token, alone
               and with the comma token blocked from the image except through the gaze heads
    route 2    every head but the gaze heads at the comma token reads the source's image keys
    layers     the comma token cannot attend to the answer token in layers 0-24, or in 20-35
    top 64     the 64 best heads of the value search cannot attend to the answer token (and 64
               random heads from their layers, drawn outside the best 128)
    8 heads    the 8 heads of the value search with the comma blocked (see 00_find_value_route_heads.py)
               read the source's values of the answer token, with and without that block

    python progress/appendix_experiments/10_value_route.py --num_pairs 500

Paper: route 1 moves the next letter in 17.0% of cases; with the comma token blocked from the image
it moves it in 99.0% (cursor 40.7%, next 58.2%). Route 2 moves the cursor in 10.8%. With the layers
0-24 or 20-35 blocked the answer survives in 96.9% and 96.9%. Blocking the 64 best heads gives 16.5%
repeats of W, against 0.0% for random heads. The 8 heads move the cursor in 63.9% with the block
(random heads 11.9%) and in 5.7% without.
These numbers came from an earlier, smaller run: 200 list pairs drawn from another seed, of which
194 count, with the head lists of utils/heads.py named below. Every held-out pair here counts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, name_answer, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import all_heads, load_model, n_layers
from utils.patching import Patch, make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("value_route", args)
    gaze = set(HEADS["gaze"])
    not_gaze = [head for head in all_heads(model) if head not in gaze]
    ranking, value_heads = HEADS["comma_reads_answer_token"], HEADS["value_route"]
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters, W = matching_letters(pair["target"]), [pair["written"]]
        names = {"own": letters[1], "cursor": letters[2], "next": letters[3], "again": pair["written"]}
        target, source = run_list(model, pair["target"], W), run_list(model, pair["source"], W)
        answer, comma = [target.token("last_answer")], target.token("last_comma")
        values = lambda heads: make_patch(source, target, "values", comma, heads=heads, positions=answer)
        gaze_view_only = lambda patch: patch.block(not_gaze, [comma], target.image_positions)
        reads_answer = lambda layers: Patch().block(None, [comma], answer, layers=layers)
        patches = {
            "route_1": values(all_heads(model)),
            "route_1_gaze_view_only": gaze_view_only(values(all_heads(model))),
            "route_2": make_patch(source, target, "image_keys", comma, heads=not_gaze),
            "layers_0_to_24_blocked": reads_answer(range(0, 25)),
            "layers_20_to_35_blocked": reads_answer(range(20, n_layers(model))),
            "top_64_blocked": Patch().block(ranking[:64], [comma], answer),
            "random_64_blocked": Patch().block(random_like(ranking[:64], seed=pair["index"], exclude=ranking[:128]), [comma], answer),
            "8_heads": values(value_heads),
            "8_heads_gaze_view_only": gaze_view_only(values(value_heads)),
            "8_random_gaze_view_only": gaze_view_only(values(random_like(value_heads, seed=pair["index"] + 7))),
        }
        said = dict(zip(patches, [name_answer(a, names) for a in run_patches(model, target, list(patches.values()))["answers"]]))
        results.add(pair["index"], said=said, **{f"{name}_{outcome}": said[name] == outcome
                                                 for name in patches for outcome in names},
                    **{f"{name}_moved": said[name] in ("cursor", "next") for name in patches})
    for condition, paper in [("route_1_moved", 17.0), ("route_1_gaze_view_only_moved", 99.0),
                             ("route_1_gaze_view_only_cursor", 40.7), ("route_1_gaze_view_only_next", 58.2),
                             ("route_2_cursor", 10.8), ("layers_0_to_24_blocked_own", 96.9),
                             ("layers_20_to_35_blocked_own", 96.9), ("top_64_blocked_again", 16.5),
                             ("random_64_blocked_again", 0.0), ("8_heads_gaze_view_only_cursor", 63.9),
                             ("8_random_gaze_view_only_cursor", 11.9), ("8_heads_cursor", 5.7)]:
        results.report(condition, paper=paper)
    results.save()


if __name__ == "__main__":
    main()
