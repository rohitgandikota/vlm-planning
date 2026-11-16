"""All conditions of the rebind (Appendix D.3, figure "rebind").

This extends progress/main_experiments/04_cursor_at_the_last_answer_token.py and
05_first_answer_token.py with the remaining head sets and outcomes, on the same pairs. The target
has its shape at panels 1, 3, 4 and 6 and has written the letters of panels 1 and 3. At one answer
token a set of heads reads the image keys of a source in which that letter sits in panel 4.
    last answer token    the letters of panels 3 and 4 swapped; the cursor moves to panel 4, so a
                         moved cursor gives the letter of panel 6
    first answer token   the letters of panels 1 and 4 swapped; the target should keep its answer
Head sets: the 32 answer-token heads, the 16 binding heads, the 18 binding, query and key heads,
all 1,152 heads, and five draws of 16 random heads from the binding heads' layers (last token only).
For each we count: the letter of panel 6 (moved), the letter of panel 4 (own answer), and a
written letter again (repeat).

    python progress/appendix_experiments/07_rebind_all_conditions.py --num_pairs 500

Paper: at the last answer token the letter of panel 6 in 88.0% (answer-token heads), 84.2% (binding
heads), 62.2% (all heads), and never with random heads. The rest still answer from panel 4 (9.8%,
13.4%, and 35.6%). A written letter is repeated in at most 2.4% of pairs in any condition. At the
first answer token the answer moves in at most 3.6% and stays at panel 4 in 96.4% to 98.8%.
These pairs are drawn for this experiment (utils.dataset.progress_pairs), so --split is not used.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import name_answer, progress_pairs, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import all_heads, load_model
from utils.patching import make_patch, run_patches

PAPER = {"last_answer_token_heads": (88.0, 9.8), "last_binding_heads": (84.2, 13.4), "last_all_heads": (62.2, 35.6),
         "first_answer_token_heads": (None, 96.4), "first_binding_heads": (None, 98.8)}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("rebind_all_conditions", args)
    sets = {"answer_token_heads": HEADS["answer_token"], "binding_heads": HEADS["binding"],
            "binding_query_key_heads": HEADS["binding_query_key"], "all_heads": all_heads(model)}
    conditions = [("last", name, heads) for name, heads in sets.items()]
    conditions += [("last", f"random_heads_{draw}", random_like(HEADS["binding"], seed=draw)) for draw in range(5)]
    conditions += [("first", "answer_token_heads", sets["answer_token_heads"]), ("first", "binding_heads", sets["binding_heads"])]
    for pair in progress_pairs(model, "cursor", args.start + args.num_pairs)[args.start:]:
        target = run_list(model, pair["target"])
        sources = {"last": run_list(model, pair["source"]), "first": run_list(model, pair["source_first"])}
        answer_token = {"first": target.token("answers")[0], "last": target.token("answers")[-1]}
        patches = [make_patch(sources[token], target, "image_keys", answer_token[token], heads=heads)
                   for token, _, heads in conditions]
        row = {}
        for (token, name, _), answer in zip(conditions, run_patches(model, target, patches)["answers"]):
            said = name_answer(answer, pair["letters"])
            row.update({f"{token}_{name}_moved": said == "after_panel_4", f"{token}_{name}_own": said == "own",
                        f"{token}_{name}_repeat": said.startswith("repeat")})
        results.add(pair["index"], **row)
    for token, name, _ in conditions:
        moved, own = PAPER.get(f"{token}_{name}", (0.0 if name.startswith("random") else None, None))
        results.report(f"{token}_{name}_moved", paper=moved if token == "last" else "at most 3.6", label=f"{token} answer token, {name}: letter of panel 6")
        results.report(f"{token}_{name}_own", paper=own, label=f"{token} answer token, {name}: letter of panel 4")
        results.report(f"{token}_{name}_repeat", paper="at most 2.4", label=f"{token} answer token, {name}: repeats a written letter")
    results.save()


if __name__ == "__main__":
    main()
