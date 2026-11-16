"""What the heads of the first selection search do (Appendix E.7, first search).

The standard list pair: the target has written W, its letter at r1; the source has the shape at r2
and r4 and wrote W at r2, so its next panel is r4. For the 64 heads found by
progress/appendix_experiments/00_find_selection_heads.py (HEADS["selection_output"]) we copy from
the source into the target at the last comma token, one part at a time:
    outputs                     the heads' outputs
    queries                     the heads' queries
    image keys                  the keys the heads see at the image tokens
    text values                 the values the heads read from the prompt and the written letter
    outputs, gaze queries held  the outputs, with the 32 gaze heads' queries held at the target's own
and the outputs of the gaze heads themselves. Random heads from the same layers are the control.
The next panel moved means the target answers its letter at r4; the cursor moved, its letter at r3.

    python progress/appendix_experiments/55_selection_heads.py --num_pairs 500

Paper: the outputs move the next panel in 61.9% of cases (random heads 0.0%). The queries move it
in 40.7%, and the cursor in 21.6%. The image keys move it in 0.0%, the text values in 5.7%. With
the gaze heads' queries held, the outputs move nothing (0.0%). Fourteen of the 64 heads are gaze
heads of that earlier run; 13 are in the current HEADS["gaze"]. The gaze heads' own outputs copy the source's letter in 100.0% of cases. These numbers come
from an earlier, smaller run (200 held-out pairs of an earlier draw of the list task, 194 valid).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    selection = HEADS["selection_output"]
    print(f"{len(set(selection) & set(HEADS['gaze']))} of the {len(selection)} selection heads are gaze heads (paper: 14, counted with the gaze heads of the earlier run)")
    results = Results("selection_heads", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])                 # the target's letters at r1, r2, r3, r4
        target = run_list(model, pair["target"], [pair["written"]])
        source = run_list(model, pair["source"], [pair["written"]])
        if target.answer != " " + letters[1] or source.answer != " " + matching_letters(pair["source"])[1]:
            continue
        text = list(range(target.token("shape"), target.token("colon") + 1)) + [target.token("last_answer")]
        hold = make_patch(target, target, "query", "last_comma", heads=HEADS["gaze"])
        names, patches = [], []
        for label, heads in (("selection", selection), ("random", random_like(selection, seed=pair["index"]))):
            for part, patch in (("outputs", make_patch(source, target, "head_output", "last_comma", heads=heads)),
                                ("queries", make_patch(source, target, "query", "last_comma", heads=heads)),
                                ("image_keys", make_patch(source, target, "image_keys", "last_comma", heads=heads)),
                                ("text_values", make_patch(source, target, "values", "last_comma", heads=heads, positions=text)),
                                ("outputs_gaze_queries_held", make_patch(source, target, "head_output", "last_comma", heads=heads).add(hold))):
                names.append(f"{label}_{part}")
                patches.append(patch)
        names.append("gaze_heads_outputs")
        patches.append(make_patch(source, target, "head_output", "last_comma", heads=HEADS["gaze"]))
        answers = dict(zip(names, run_patches(model, target, patches)["answers"]))
        row = {f"{name}_next_panel": answer == " " + letters[3] for name, answer in answers.items()}
        row.update({f"{name}_cursor": answer == " " + letters[2] for name, answer in answers.items()})
        row["gaze_heads_outputs_copy_source_letter"] = answers["gaze_heads_outputs"] == source.answer
        results.add(pair["index"], **row)
    for condition, paper in (("selection_outputs_next_panel", 61.9), ("random_outputs_next_panel", 0.0),
                             ("selection_queries_next_panel", 40.7), ("selection_queries_cursor", 21.6),
                             ("selection_image_keys_next_panel", 0.0), ("selection_text_values_next_panel", 5.7),
                             ("selection_outputs_gaze_queries_held_next_panel", 0.0), ("gaze_heads_outputs_copy_source_letter", 100.0)):
        results.report(condition, paper=paper, label=condition.replace("_", " "))
    results.save()


if __name__ == "__main__":
    main()
