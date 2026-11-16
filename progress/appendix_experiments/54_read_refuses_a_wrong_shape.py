"""The read refuses a written panel and a wrong shape (Appendix E.6).

This extends progress/main_experiments/10_gaze_heads_sent_to_written_panel.py. The target has
written W, its letter at r1. We copy the queries of the 32 gaze heads at the last comma token from a
source whose next answer sits at a chosen panel, and ask whether the target's answer goes there:
    another shape    "source_other": the shape at r1 (holding W) and at the first later panel that
                     holds another shape in the target
    written panel    "source_written": the shape at r1 and at the panel just before it (holding W),
                     so its next panel, r1, is already written in the target
For each source we run the queries alone, and with the gaze heads also reading the source's image
keys at the comma token. For the written panel we also block, on top of the queries, the comma
token from the answer token of W (in every layer and head), and the answer token of W from the
image (in every layer and head).

    python progress/appendix_experiments/54_read_refuses_a_wrong_shape.py --num_pairs 500

Paper: another shape 22.4% (85.1% with the source's image keys). Written panel 0.6% (22.0% with the
keys); with the comma token blocked from the answer token 97.6% (100.0% with the keys as well);
with the answer token blocked from the image 6.1%. The sources exist for 464 and 164 of the 500
held-out pairs, and every one of those pairs counts (both runs give their own next letter).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model, n_layers
from utils.patching import Patch, make_patch, run_patches

SOURCES = {"another_shape": "source_other", "written_panel": "source_written"}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    every_layer, gaze = range(n_layers(model)), HEADS["gaze"]
    results = Results("read_refuses_a_wrong_shape", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        target = run_list(model, pair["target"], [pair["written"]])
        if target.answer != " " + matching_letters(pair["target"])[1]:
            continue
        comma, letter = target.token("last_comma"), target.token("last_answer")
        queries = lambda: make_patch(source, target, "query", "last_comma", heads=gaze)
        keys = lambda: make_patch(source, target, "image_keys", "last_comma", heads=gaze)
        no_letter = lambda: Patch().block(None, [comma], [letter], layers=every_layer)      # the comma cannot attend to W
        letter_blind = lambda: Patch().block(None, [letter], target.image_positions, layers=every_layer)
        case = {}
        for name, side in SOURCES.items():
            if side not in pair:
                continue
            source = run_list(model, pair[side], [pair["written"]])
            panel = pair[side]["matching"][1]                       # the panel of the source's next answer
            if source.answer != " " + pair[side]["letters"][panel]:
                continue
            patches = {"queries": queries(), "queries_and_keys": queries().add(keys()),
                       "random_heads_queries": make_patch(source, target, "query", "last_comma", heads=random_like(gaze, pair["index"]))}
            if name == "written_panel":
                patches.update({"queries_comma_blocked_from_W": queries().add(no_letter()),
                                "queries_and_keys_comma_blocked_from_W": queries().add(keys()).add(no_letter()),
                                "queries_W_blind": queries().add(letter_blind())})
            there = " " + pair["target"]["letters"][panel]         # the target's letter at that panel
            answers = run_patches(model, target, list(patches.values()))["answers"]
            case.update({f"{name}_{arm}": answer == there for arm, answer in zip(patches, answers)})
        results.add(pair["index"], **case)
    for condition, paper in (("another_shape_queries", 22.4), ("another_shape_queries_and_keys", 85.1),
                             ("another_shape_random_heads_queries", None), ("written_panel_queries", 0.6),
                             ("written_panel_queries_and_keys", 22.0), ("written_panel_queries_comma_blocked_from_W", 97.6),
                             ("written_panel_queries_and_keys_comma_blocked_from_W", 100.0), ("written_panel_queries_W_blind", 6.1),
                             ("written_panel_random_heads_queries", None)):
        results.report(condition, paper=paper, label=condition.replace("_", " ") + ": answer goes there")
    results.save()


if __name__ == "__main__":
    main()
