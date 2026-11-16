"""The gaze heads do not read a written panel (Section 5.4).

The target has written one letter, W, from its first matching panel. We copy the queries of the 32
gaze heads at the last comma token from a source whose next answer sits at a chosen panel. This
points the gaze heads of the target at that panel.
    unwritten panel   the source's next panel is the target's fourth matching panel. The target
                      has not written it, so it should answer with its letter there.
    written panel     the source's next panel is the target's first matching panel, which the
                      target has already written. If the read refuses a written panel, the target
                      should not write W again.

    python progress/main_experiments/10_gaze_heads_sent_to_written_panel.py --num_pairs 500

Paper: the answer moves to the written panel in 0.6% of cases, and to the unwritten panel in 88.2%.
The written-panel source needs a panel to the left of the target's first matching panel, so it
exists for 164 of the 500 held-out pairs. The extra source images exist for held-out pairs only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches

SOURCES = {"unwritten_panel": "source", "written_panel": "source_written"}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("gaze_heads_sent_to_written_panel", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        written = [pair["written"]]
        target = run_list(model, pair["target"], written)
        random_heads = random_like(HEADS["gaze"], seed=pair["index"])
        case = {}
        for name, side in SOURCES.items():
            if side not in pair:
                continue
            source = run_list(model, pair[side], written)
            panel = pair[side]["matching"][1]                       # the panel of the source's next answer
            # Both runs must give their own next letter with no patch.
            if target.answer != " " + matching_letters(pair["target"])[1] or source.answer != " " + pair[side]["letters"][panel]:
                continue
            patches = [make_patch(source, target, "query", "last_comma", heads=heads) for heads in (HEADS["gaze"], random_heads)]
            gaze, control = run_patches(model, target, patches)["answers"]
            there = " " + pair["target"]["letters"][panel]         # the target's letter at that panel
            case.update({name: gaze == there, name + "_random_heads": control == there, name + "_answer": gaze})
        results.add(pair["index"], **case)
    results.report("written_panel", paper=0.6, label="sent to a written panel: answer moves there")
    results.report("unwritten_panel", paper=88.2, label="sent to an unwritten panel: answer moves there")
    results.report("written_panel_random_heads", label="random heads, written panel")
    results.report("unwritten_panel_random_heads", label="random heads, unwritten panel")
    results.save()


if __name__ == "__main__":
    main()
