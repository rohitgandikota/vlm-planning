"""On a blank canvas the binding heads set the cursor, with or without the layout filter (Appendix G.3.2).

The list pairs on canvases use only layouts with a clear reading order. To see whether progress
needs this, we repeat the test of progress/main_experiments/04 on canvases, and take the order of
the four asked shapes, t1 to t4, from the model's own list, not from a rule. Two groups of layouts:
    clear     the reading order is the same for row heights 60, 160, 260 for every group of shapes
    unclear   the reading-order rule gives different orders at these row heights for the asked shapes
The target has written the letters of t1 and t2 and should answer t3. In the source, the image is
the same with the letters of t2 and t3 swapped, so the last written letter sits at t3. At the last
answer token, the 16 binding heads read the source's image keys. If they set the cursor, the target
skips t3 and answers t4. As a control, the same patch at the first answer token, from the image with
the letters of t1 and t3 swapped, should leave the answer unchanged.

    python generalization/main_experiments/28_canvas_cursor_without_layout_filter.py

--num_pairs is the number of pairs per group (the paper used 50). A layout is used when the model's
own list names each asked shape exactly once and then stops, and the target with two letters
written answers t3. The random heads are one fixed draw, random_like(binding, seed=5).

Paper: the next answer moves to t4 in 70.0% of pairs on clear layouts and 82.0% on unclear ones;
random heads 0.0% and 2.0%. The patch at the first answer token leaves 94.0% and 82.0% of answers
unchanged, and the model repeats a written letter in at most 8.0% of pairs.
"""
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import (LETTERS, LIST_ASK, clear_order, draw_canvas, list_layout_side, list_prefill, list_question,
                           name_list_tokens, sample_centers)
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_model, run_patches, write_tokens

GROUPS = {"clear": 20261115, "unclear": 20261116}         # the seed of each group


def sample_layout(rng, group):
    """Six centers on a 960 x 960 canvas and the four asked places, from the given group."""
    while True:
        centers, places = sample_centers(rng, 6, 960), sorted(rng.sample(range(6), 4))
        if group == "clear" and all(clear_order(centers, list(g)) is not None
                                    for k in range(2, 7) for g in itertools.combinations(range(6), k)):
            return centers, places
        if group == "unclear" and clear_order(centers, places) is None:
            return centers, places


def run_canvas(model, side, centers, letters, written):
    image = draw_canvas(side["shapes"], letters, centers)
    run = run_model(model, image, list_question(side["ask"]), list_prefill(written))
    return name_list_tokens(run, side, written, end_on_letter=False)


def main():
    args = parse_args(__doc__, num_pairs=50)
    model = load_model(args.model, args.device)
    random_heads = random_like(HEADS["binding"], seed=5)
    results = Results("canvas_cursor_without_layout_filter", args)
    for group, seed in GROUPS.items():
        rng, kept, tries = random.Random(seed), 0, 0
        while kept < args.num_pairs and tries < 6 * args.num_pairs:
            tries += 1
            centers, places = sample_layout(rng, group)
            side = list_layout_side(rng, rng.choice(LIST_ASK), places, [""] * 6)
            L = rng.sample(LETTERS, 6)
            unpatched = run_model(model, draw_canvas(side["shapes"], L, centers), list_question(side["ask"]), record=False)
            own_list = [item.strip() for item in write_tokens(model, unpatched, max_tokens=8).split(",") if item.strip()]
            if sorted(own_list) != sorted(L[p] for p in places) or len(own_list) != 4:
                continue
            t1, t2, t3, t4 = (L.index(letter) for letter in own_list)
            written = [L[t1], L[t2]]
            target = run_canvas(model, side, centers, L, written)
            if target.answer != " " + L[t3]:
                continue
            swapped = lambda a, b: [L[b] if p == a else L[a] if p == b else L[p] for p in range(6)]
            last, first = run_canvas(model, side, centers, swapped(t2, t3), written), run_canvas(model, side, centers, swapped(t1, t3), written)
            answers = run_patches(model, target, [
                make_patch(last, target, "image_keys", "last_answer", heads=HEADS["binding"]),
                make_patch(last, target, "image_keys", "last_answer", heads=random_heads),
                make_patch(first, target, "image_keys", target.token("answers")[0], heads=HEADS["binding"])])["answers"]
            results.add(kept, group=group, **{f"{group}_moves_to_t4": answers[0] == " " + L[t4],
                                              f"{group}_random_heads": answers[1] == " " + L[t4],
                                              f"{group}_first_answer_token_unchanged": answers[2] == " " + L[t3],
                                              f"{group}_repeats": answers[0] in (" " + L[t1], " " + L[t2])})
            kept += 1
        print(f"{group} layouts: {kept} pairs in {tries} tries", flush=True)
    for group, paper in (("clear", (70.0, 0.0, 94.0)), ("unclear", (82.0, 2.0, 82.0))):
        results.report(f"{group}_moves_to_t4", paper=paper[0], label=f"{group} layouts: binding heads, answer moves to t4")
        results.report(f"{group}_random_heads", paper=paper[1], label=f"{group} layouts: random heads, answer moves to t4")
        results.report(f"{group}_first_answer_token_unchanged", paper=paper[2], label=f"{group} layouts: first answer token, unchanged")
        results.report(f"{group}_repeats", paper="at most 8.0", label=f"{group} layouts: repeats a written letter")
    results.save()


if __name__ == "__main__":
    main()
