"""Every blocking experiment in the target's own run (Appendix C.3, Table "knockouts-full").

No source run is used. In the target's own run we block attention from some tokens to some
positions, and count the cases that keep the target's own answer T ("letter"), and the cases in
which the published Gaze Heads, at the colon token, still look most at the panel of T ("gaze").
The row "relay heads blocked" is relay/main_experiments/03_relay_heads_blocked.py.

When blocking breaks the pointer, the wrong answers lean toward the leftmost panel. Over the two
conditions that block the pointer (later tokens cannot attend to the shape token, and relay heads
blocked) we count the wrong answers that are the target's letter in the leftmost panel.

    python relay/appendix_experiments/04_blocked_attention_all_conditions.py --num_pairs 500

Paper (letter / gaze, percent of cases): nothing blocked 100.0 / 97.4; query and key heads 85.3 /
81.6 (random 100.0 / 97.7); binding, query and key heads 84.9 / 74.7; every head at the shape
token 40.4 / 35.9; shape token to the token before the colon token 32.6 / 30.0 (control 99.3 /
99.6); only the shape token sees the image 5.3 / 24.6; later tokens cannot attend to the shape
token 25.3 / 26.5 (control 99.9 / 97.5); only the colon token 99.7 / 97.2; only the tokens in
between 90.6 / 92.4; relay heads 7.8 / 19.1 (random 99.8 / 97.9). Leftmost panel: 39.8% of the
wrong answers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model, n_layers
from utils.patching import Patch, panel_attention, run_patches

PAPER = {"nothing_blocked": (100.0, 97.4), "query_and_key_heads": (85.3, 81.6), "query_and_key_random": (100.0, 97.7),
         "binding_query_key_heads": (84.9, 74.7), "every_head_at_shape": (40.4, 35.9),
         "shape_to_before_colon": (32.6, 30.0), "control_text_before_shape": (99.3, 99.6),
         "only_shape_sees_image": (5.3, 24.6), "later_cannot_attend_shape": (25.3, 26.5),
         "control_later_cannot_attend_previous": (99.9, 97.5), "only_colon_cannot_attend_shape": (99.7, 97.2),
         "only_between_cannot_attend_shape": (90.6, 92.4), "relay_heads": (7.8, 19.1), "relay_random": (99.8, 97.9)}
POINTER_BROKEN = ("later_cannot_attend_shape", "relay_heads")       # the two conditions of the leftmost-panel count


def conditions(model, target, seed):
    """Every blocking condition of the table, as {name: patch}."""
    layers = range(n_layers(model))
    image, first = target.image_positions, target.first_text
    shape, colon = target.token("shape"), target.token("colon")
    between = range(shape + 1, colon)                               # the tokens after the shape token, before the colon token
    after = range(shape + 1, colon + 1)                             # every token after the shape token
    every = lambda tokens, positions: Patch().block(None, tokens, positions, layers=layers)
    relay_text = range(shape, colon)                                # the shape token through the token before the colon token
    return {
        "nothing_blocked": Patch(),
        "query_and_key_heads": Patch().block(HEADS["query_and_key"], [shape], image),
        "query_and_key_random": Patch().block(random_like(HEADS["query_and_key"], seed), [shape], image),
        "binding_query_key_heads": Patch().block(HEADS["binding_query_key"], [shape], image),
        "every_head_at_shape": every([shape], image),
        "shape_to_before_colon": every(range(shape, colon), image),
        "control_text_before_shape": every(range(first, shape), image),
        "only_shape_sees_image": every([t for t in range(first, colon + 1) if t != shape], image),
        "later_cannot_attend_shape": every(after, [shape]),
        "control_later_cannot_attend_previous": every(after, [shape - 1]),
        "only_colon_cannot_attend_shape": every([colon], [shape]),
        "only_between_cannot_attend_shape": every(between, [shape]),
        "relay_heads": Patch().block(HEADS["relay"], [colon], relay_text),
        "relay_random": Patch().block(random_like(HEADS["relay"], seed), [colon], relay_text)}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("blocked_attention_all_conditions", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for _, target, letters in both_directions(model, pair):     # each side of the pair is a target once
            patches = conditions(model, target, seed=1000 + pair["index"])
            result = run_patches(model, target, list(patches.values()), keep_attention=True)
            looked_at = panel_attention(result["attention"], HEADS["published_gaze"], target).argmax(dim=1).tolist()
            own_panel = target.image.letters.index(letters["T"])
            leftmost = " " + target.image.letters[0]
            row, wrong = {}, []
            for name, answer, panel in zip(patches, result["answers"], looked_at):
                row[f"{name}_letter"] = answer == " " + letters["T"]
                row[f"{name}_gaze"] = panel == own_panel
                if name in POINTER_BROKEN and answer != " " + letters["T"]:
                    wrong.append((name, answer))
            results.add(pair["index"], **row)
            # the leftmost-panel share counts wrong answers, not cases: one extra row per wrong answer
            for name, answer in wrong:
                results.add(pair["index"], condition=name, wrong_answer_leftmost=answer == leftmost)
    for name, (letter, gaze) in PAPER.items():
        results.report(f"{name}_letter", paper=letter)
        results.report(f"{name}_gaze", paper=gaze)
    results.report("wrong_answer_leftmost", paper=39.8, label="wrong answers that are the leftmost panel's letter")
    results.save()


if __name__ == "__main__":
    main()
