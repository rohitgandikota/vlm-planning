"""Every list-task search finds mostly the heads of Sections 4 and 6 (Appendix D.3, Table "many heads").

Each row is one patch on the held-out list pairs, made three times: with the heads of Sections 4
and 6 unchanged, with random heads from their layers, and with the heads a new search on the list
task found for this patch (best first, K heads). A case counts when the target answers with the
letter that the moved variable predicts.
    first answer   nothing written; the source shows the asked shape at r2 and r4, so a moved
                   variable gives the target's letter at r2
    next answer    W written; with the answer token's image keys the cursor moves from r1 to r2
                   (letter at r3); with the comma token's queries the next panel moves to r4

    python progress/appendix_experiments/06_list_task_heads.py --num_pairs 500

Paper (heads of Sections 4 and 6 / random / found):
    first answer, values read at the colon token        78.5 / 0.0 / 83.0 (16)
    first answer, shape token outputs                   53.4 / 0.5 / 74.8 (32)
    first answer, shape token image key reads            9.9 / 0.0 / 10.8 (128)
    first answer, colon token queries                   46.1 / 0.0 / 57.8 (128)
    last comma token, queries (next panel moves)        88.2 / 0.0 / 86.0 (16)
    answer token, image key reads (the text)              -  / 0.0 / 73.8 (32)
Overlaps: 15 of the 32 shape-token heads are binding heads, 10 of the 16 colon-token heads are
relay heads, 15 of the 16 comma-token heads are gaze heads, and the 64 stop heads hold 1 binding,
10 relay, 6 gaze, and 0 language heads.
The first-answer values with the heads of Sections 4 and 6, and their random heads, came from an
earlier, smaller run with earlier head sets (all but 88.2 in that column). The random heads of the
found sets are drawn here outside the K found heads; the paper drew them outside the best 2K.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches


def colon_values(source, target, heads):
    question = range(target.token("shape"), target.token("colon"))      # the shape token up to the colon token
    return make_patch(source, target, "values", "colon", heads=heads, positions=question)


# name: (answer, patch, heads of Sections 4 and 6 or None, found heads, paper values)
ROWS = {
    "first_colon_values": ("first", colon_values, "relay", "list_colon_values", (78.5, 0.0, 83.0)),
    "first_shape_outputs": ("first", lambda s, t, h: make_patch(s, t, "head_output", "shape", heads=h),
                            "binding", "list_shape_token", (53.4, 0.5, 74.8)),
    "first_shape_keys": ("first", lambda s, t, h: make_patch(s, t, "image_keys", "shape", heads=h),
                         "key", "list_shape_token_keys", (9.9, 0.0, 10.8)),
    "first_colon_queries": ("first", lambda s, t, h: make_patch(s, t, "query", "colon", heads=h),
                            "gaze", "list_colon_queries", (46.1, 0.0, 57.8)),
    "next_comma_queries": ("next", lambda s, t, h: make_patch(s, t, "query", "last_comma", heads=h),
                           "gaze", "list_comma_queries", (88.2, 0.0, 86.0)),
    "next_answer_keys": ("next", lambda s, t, h: make_patch(s, t, "image_keys", "last_answer", heads=h),
                         None, "answer_token", (None, 0.0, 73.8)),
}


def print_overlaps():
    for found, size, theirs in (("list_shape_token", 32, "binding_query_key"), ("list_colon_values", 16, "relay"),
                                ("list_comma_queries", 16, "gaze")):
        print(f"{len(set(HEADS[found]) & set(HEADS[theirs]))} of the {size} heads of {found} are in {theirs}")
    stop = set(HEADS["stop"])
    print("stop heads: " + ", ".join(f"{len(stop & set(HEADS[name]))} {name}" for name in ("binding_query_key", "relay", "gaze", "language")))
    print("paper: 15 of 32, 10 of 16, 15 of 16; stop heads 1 binding, 10 relay, 6 gaze, 0 language\n")


def main():
    args = parse_args(__doc__)
    print_overlaps()
    model = load_model(args.model, args.device)
    results = Results("list_task_heads", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters, W = matching_letters(pair["target"]), [pair["written"]]
        runs = {"first": (run_list(model, pair["source_first"], []), run_list(model, pair["target"], [])),
                "next": (run_list(model, pair["source"], W), run_list(model, pair["target"], W))}
        expected = {"first": letters[1], "next_comma_queries": letters[3], "next_answer_keys": letters[2]}
        row = {}
        for name, (answer, build, theirs, found, _) in ROWS.items():
            source, target = runs[answer]
            sets = {"found": HEADS[found], "found_random": random_like(HEADS[found], seed=pair["index"])}
            if theirs:
                sets.update(theirs=HEADS[theirs], theirs_random=random_like(HEADS[theirs], seed=pair["index"]))
            answers = run_patches(model, target, [build(source, target, heads) for heads in sets.values()])["answers"]
            want = expected[name] if name in expected else expected[answer]
            row.update({f"{name}_{kind}": said == " " + want for kind, said in zip(sets, answers)})
        results.add(pair["index"], **row)
    for name, (_, _, theirs, found, (paper_theirs, paper_random, paper_found)) in ROWS.items():
        if theirs:
            results.report(f"{name}_theirs", paper=paper_theirs, label=f"{name}: heads of Sections 4 and 6")
            results.report(f"{name}_theirs_random", paper=paper_random, label=f"{name}: random heads")
        results.report(f"{name}_found", paper=paper_found, label=f"{name}: {len(HEADS[found])} heads found here")
        results.report(f"{name}_found_random", paper=paper_random if not theirs else None, label=f"{name}: random heads for the found set")
    results.save()


if __name__ == "__main__":
    main()
