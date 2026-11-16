"""Create the images, questions, and source-target pairs of the paper.

Everything here is plain Python and PIL, so you can build your own data:

    from utils.dataset import draw_strip, one_answer_question
    image = draw_strip(shapes=["star", "square", "circle", "heart"], letters=["A", "B", "C", "D"])
    question = one_answer_question("circle")          # "... What is the letter inside the circle?"

The paper's own pairs come from the functions further down. They are generated from a seed, and a
pair is kept only if the model answers both of its questions correctly with no patch.
"""
import itertools
import json
import math
import os
import random
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import collections
import string

FONT = Path(__file__).parent / "fonts" / "DejaVuSans-Bold.ttf"
CACHE = Path(__file__).resolve().parents[1] / "dataset"


def cache_path(model, name):
    """The file that keeps the no-patch checks of one task. Each model has its own files: the 8B's
    are in dataset/, another model's are in dataset/<model name>/."""
    model_id = getattr(model, "model_id", "Qwen/Qwen3-VL-8B-Instruct")
    folder = CACHE if model_id == "Qwen/Qwen3-VL-8B-Instruct" else CACHE / model_id.split("/")[-1]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name

SHAPES = ("star", "square", "circle", "triangle", "diamond", "plus", "heart")
LETTERS = tuple("ABCDEFGHKMNPRSTUVWXYZ")      # Q, O, I, L, J are left out: the model misreads them
PANEL = 384                                   # each panel is 384 x 384 pixels

ONE_ANSWER_PROMPT = "Answer with the letter only. What is the letter inside the {shape}?"


# ------------------------------------------------------------------------------------ drawing
def _polygon(cx, cy, radius, sides):
    return [(cx + radius * math.cos(-math.pi / 2 + 2 * math.pi * i / sides),
             cy + radius * math.sin(-math.pi / 2 + 2 * math.pi * i / sides)) for i in range(sides)]


def _star(cx, cy, radius):
    return [(cx + (radius if i % 2 == 0 else radius * 0.44) * math.cos(-math.pi / 2 + i * math.pi / 5),
             cy + (radius if i % 2 == 0 else radius * 0.44) * math.sin(-math.pi / 2 + i * math.pi / 5)) for i in range(10)]


def _heart(cx, cy, radius):
    points = []
    for i in range(160):
        t = 2 * math.pi * i / 160
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        points.append((cx + radius * x / 18, cy - radius * y / 18 + radius * 0.08))
    return points


def _plus(cx, cy, r, t):
    return [(cx - t, cy - r), (cx + t, cy - r), (cx + t, cy - t), (cx + r, cy - t), (cx + r, cy + t), (cx + t, cy + t),
            (cx + t, cy + r), (cx - t, cy + r), (cx - t, cy + t), (cx - r, cy + t), (cx - r, cy - t), (cx - t, cy - t)]


# The paper draws the shapes in two sizes: "small" in the one-answer task and "large" in the list
# task. Each entry: radius, vertical shift of the shape, vertical shift of the letter.
_SIZES = {
    "small": {"circle": (53, 0, 1), "square": (53, 0, 1), "triangle": (53, 4, 10), "diamond": (53, 0, 1),
              "plus": (53, 0, 1), "heart": (53, 0, 1), "star": (70, 4, 6)},
    "large": {"circle": (72, 0, 1), "square": (68, 0, 1), "triangle": (96, 10, 22), "diamond": (88, 0, 1),
              "plus": (88, 0, 1), "heart": (92, 0, 16), "star": (100, 4, 8)},
}


def draw_shape(draw, shape, letter, cx, cy, font, size="small"):
    """Draw one outlined shape with one letter inside it, centered at (cx, cy)."""
    r, shape_dy, letter_dy = _SIZES[size][shape]
    style = {"fill": "white", "outline": "black", "width": 6}
    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), **style)
    elif shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), **style)
    elif shape == "triangle":
        draw.polygon(_polygon(cx, cy + shape_dy, r, 3), **style)
    elif shape == "diamond":
        draw.polygon(_polygon(cx, cy, r, 4), **style)
    elif shape == "star":
        draw.polygon(_star(cx, cy + shape_dy, r), **style)
    elif shape == "plus":
        draw.polygon(_plus(cx, cy, r, r * (0.36 if size == "small" else 0.42)), **style)
    elif shape == "heart":
        draw.polygon(_heart(cx, cy, r), **style)
    else:
        raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")
    outline = {"stroke_width": 1, "stroke_fill": "white"} if size == "small" else {}
    draw.text((cx, cy + letter_dy), letter, font=font, fill="black", anchor="mm", **outline)


def draw_strip(shapes, letters, centers=None, size="small"):
    """A row of blank panels, one shape with one letter in each. Returns a PIL image.

    shapes, letters   lists of the same length, one entry per panel
    centers           optional (x, y) of each shape inside its panel; default is the panel center
    size              "small" (one-answer task) or "large" (list task) shapes

    The image carries `image.regions` (the box of each panel), `image.shapes` and `image.letters`."""
    assert len(shapes) == len(letters)
    font = ImageFont.truetype(str(FONT), 48)
    image = Image.new("RGB", (len(shapes) * PANEL, PANEL), "white")
    draw = ImageDraw.Draw(image)
    for i, (shape, letter) in enumerate(zip(shapes, letters)):
        x0 = i * PANEL
        draw.rectangle((x0, 0, x0 + PANEL - 1, PANEL - 1), outline="#666666", width=3)
        cx, cy = (PANEL // 2, PANEL // 2) if centers is None else centers[i]
        draw_shape(draw, shape, letter, x0 + cx, cy, font, size)
    image.regions = [[i * PANEL, 0, (i + 1) * PANEL, PANEL] for i in range(len(shapes))]
    image.shapes, image.letters = list(shapes), list(letters)
    return image


def one_answer_question(shape):
    return ONE_ANSWER_PROMPT.format(shape=shape)


# ------------------------------------------------------------------------------------ the one-answer task
ONE_ANSWER_SEED = 20260930
SPLITS = {"search": (0, 200), "validation": (200, 300), "heldout": (300, 800)}   # ranks among the valid pairs


def sample_one_answer_pair(candidate, seed=ONE_ANSWER_SEED):
    """One source-target pair of the one-answer task (Section 3 of the paper).

    Both images hold the same four shapes in different orders, and eight different letters. The two
    questions ask for different shapes. The layout is chosen so that the panels below are all
    different, which makes each possible effect of a patch give a different letter (see
    `outcome_letters`). Returns {"candidate", "source", "target"}; each side is a dict with
    "shapes", "letters", "ask" (the shape asked for), "centers"."""
    rng = random.Random(seed + candidate)
    shapes = rng.sample(SHAPES, 4)
    source_ask, target_ask = rng.sample(shapes, 2)
    source_order = rng.sample(shapes, 4)
    free = [i for i in range(4) if i not in (source_order.index(source_ask), source_order.index(target_ask))]
    rng.shuffle(free)
    target_order = [None] * 4
    target_order[free[0]] = source_ask
    target_order[free[1]] = target_ask
    others = [s for s in shapes if s not in (source_ask, target_ask)]
    rng.shuffle(others)
    for i, shape in zip([i for i, s in enumerate(target_order) if s is None], others):
        target_order[i] = shape
    letters = rng.sample(LETTERS, 8)
    centers = [(rng.randint(78, 306), rng.randint(78, 306)) for _ in range(4)]
    return {"candidate": candidate,
            "source": {"shapes": source_order, "letters": letters[:4], "ask": source_ask, "centers": centers},
            "target": {"shapes": target_order, "letters": letters[4:], "ask": target_ask, "centers": centers}}


def outcome_letters(source, target):
    """The letter the target would answer under each possible effect of a patch.

        T  the target's own answer (nothing changed)
        D  the source's answer (the answer itself was copied)
        K  the target's letter at the panel the source points to (the pointer moved)
        P  the target's letter inside the shape the source asks for (the request moved)
        R  the target's letter at the panel where the source image holds the target's shape (the layout moved)
        C  the source's letter at the panel the target points to (the image content moved)
    All six letters are different for every pair."""
    sp, tp = source["shapes"].index, target["shapes"].index
    return {"T": target["letters"][tp(target["ask"])],
            "D": source["letters"][sp(source["ask"])],
            "K": target["letters"][sp(source["ask"])],
            "P": target["letters"][tp(source["ask"])],
            "R": target["letters"][sp(target["ask"])],
            "C": source["letters"][tp(target["ask"])]}


OUTCOMES = {"T": "the target's own answer (nothing changed)",
            "D": "the source's answer (the answer itself was copied)",
            "K": "the target's letter at the panel the source points to (the pointer moved)",
            "P": "the target's letter inside the shape the source asks for (the request moved)",
            "R": "the target's letter where the source image holds the target's shape (the layout moved)",
            "C": "the source's letter at the panel the target points to (the image content moved)"}


def explain(answer, letters):
    """Say which outcome an answer token is, for example ' S' -> 'K: ... (the pointer moved)'."""
    for name, letter in letters.items():
        if answer == " " + letter:
            return f"{name}: {OUTCOMES[name]}"
    return "another token"


def one_answer_image(side):
    return draw_strip(side["shapes"], side["letters"], side.get("centers"), size="small")


def one_answer_pairs(model, split="heldout", num_pairs=None, seed=ONE_ANSWER_SEED):
    """The paper's pairs for one split: "search" (200 pairs), "validation" (100) or "heldout" (500).

    Candidate pairs are generated in order from the seed. A candidate is kept if the model answers
    both of its questions correctly with no patch. The kept pairs are numbered in order, and the
    splits are ranges of that numbering, so the same seed always gives the same pairs. The checks
    are saved in dataset/, so they run only once. `num_pairs` takes the first pairs of the split."""
    from utils.patching import run_model                      # imported here so drawing needs no GPU
    first, last = SPLITS[split]
    if num_pairs is not None:
        last = min(last, first + num_pairs)
    cache_file = cache_path(model, f"one_answer_seed{seed}.json")
    checked = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    pairs, candidate = [], 0
    while len(pairs) < last:
        if str(candidate) not in checked:
            pair = sample_one_answer_pair(candidate, seed)
            checked[str(candidate)] = all(
                run_model(model, one_answer_image(side), one_answer_question(side["ask"]), record=False).answer
                == " " + side["letters"][side["shapes"].index(side["ask"])]
                for side in (pair["source"], pair["target"]))
            CACHE.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps(checked))
        if checked[str(candidate)]:
            pair = sample_one_answer_pair(candidate, seed)
            pair["index"] = len(pairs)                         # the pair's number among the kept pairs
            pairs.append(pair)
        candidate += 1
    return pairs[first:last]


def run_one_answer(model, side):
    """Run the model on one side of a pair and name its two key tokens: "shape" and "colon"."""
    from utils.patching import run_model
    run = run_model(model, one_answer_image(side), one_answer_question(side["ask"]))
    run.name_token("shape", run.find(" " + side["ask"]))
    run.name_token("colon", run.last)
    return run


def both_directions(model, pair):
    """Run both sides of a pair once, then use each side as the source of the other.

    Yields (source_run, target_run, letters) twice, where `letters` are the outcome letters."""
    a, b = run_one_answer(model, pair["source"]), run_one_answer(model, pair["target"])
    yield a, b, outcome_letters(pair["source"], pair["target"])
    yield b, a, outcome_letters(pair["target"], pair["source"])


def outcome_of(answer, letters):
    """Name the outcome of a patched run as one character.

    `answer` is the model's next token as text and `letters` comes from `outcome_letters`. Returns
    "T", "D", "K", "P", "R" or "C" when the answer is that letter, and "-" for anything else."""
    for name, letter in letters.items():
        if answer == " " + letter:
            return name
    return "-"


# ------------------------------------------------------------------------------------ the list task
# One row of six panels. Several panels hold the shape we ask for, and the model lists their letters:
# "Answer: T, W, E". We can write the first letters ourselves and read the model's next token.
LIST_PROMPT = "Answer with the letters only, separated by a comma and a space. What is the letter inside each {shape}?"
LIST_ASK = ("star", "heart", "triangle")          # the shapes the model lists reliably
# Shapes the model confuses with the shape we ask for. They never share an image with it.
CONFUSED = {"star": {"triangle", "plus"}, "triangle": {"diamond", "heart"}, "heart": set()}
LIST_SEED = 20260931


def list_question(shape):
    return LIST_PROMPT.format(shape=shape)


def list_prefill(written=(), end_on_letter=False):
    """The start of the model's answer, with the letters we write for it.

        list_prefill()                               "Answer:"          next token: the first letter
        list_prefill(["T"])                          "Answer: T,"       next token: the second letter
        list_prefill(["T", "W"])                     "Answer: T, W,"
        list_prefill(["T", "W"], end_on_letter=True) "Answer: T, W"     next token: a comma, or the end"""
    text = "Answer:" + ",".join(f" {letter}" for letter in written)
    return text + ("," if written and not end_on_letter else "")


def list_side(rng, ask, matching, letters, centers=None, written=(), other_shapes=SHAPES):
    """One image of the list task, as a dict. `matching` are the panels (0 to 5) that hold the shape
    we ask for. The other panels get a random shape from `other_shapes`, which may repeat."""
    others = [s for s in other_shapes if s != ask and s not in CONFUSED[ask]]
    shapes = [ask if panel in matching else rng.choice(others) for panel in range(6)]
    return {"shapes": shapes, "letters": list(letters), "ask": ask, "matching": sorted(matching),
            "centers": centers, "written": list(written)}


def list_image(side):
    """Draw one side. The list task uses the large shapes. With no centers they sit at the panel center."""
    return draw_strip(side["shapes"], side["letters"], side["centers"], size="large")


def matching_letters(side):
    """The letters of the matching panels from left to right: the list the model should write."""
    return [side["letters"][panel] for panel in side["matching"]]


def name_answer(answer, letters):
    """Which of the named letters an answer is. `letters` maps a name to a letter. Returns "other" for none."""
    return next((name for name, letter in letters.items() if answer.strip() == letter), "other")


def run_list(model, side, written=None, end_on_letter=False, record=True):
    """Run the model on one side of a list pair and name its tokens.

    `written` are the letters we write after "Answer:" (default: side["written"]). The named tokens:
        "shape"                      the shape token of the question
        "colon"                      the colon token of "Answer:"
        "answers", "commas"          lists: the position of every written letter and of every comma
        "last_answer", "last_comma"  the last of each (only when a letter is written)
    The model's next token is read after `run.last`: the colon token, the last comma token, or with
    `end_on_letter` the last answer token."""
    from utils.patching import run_model                      # imported here so drawing needs no GPU
    written = side["written"] if written is None else list(written)
    run = run_model(model, list_image(side), list_question(side["ask"]), list_prefill(written, end_on_letter), record=record)
    colon = run.last - 2 * len(written) + (1 if written and end_on_letter else 0)
    assert run.decode(run.ids[colon]) == ":", "the answer letters must be one token each"
    answers = [colon + 1 + 2 * i for i in range(len(written))]
    commas = [position + 1 for position in answers if position + 1 <= run.last]
    run.name_token("shape", run.find(" " + side["ask"]))
    run.name_token("colon", colon).name_token("answers", answers).name_token("commas", commas)
    if answers:
        run.name_token("last_answer", answers[-1])
    if commas:
        run.name_token("last_comma", commas[-1])
    return run


# ------------------------------------------------------------------------------------ the standard list pairs
def list_candidates(seed=LIST_SEED):
    """Candidate pairs of the list task, generated in order from one seed, without end.

    A pair has one target and several sources that share the shape we ask for and the shape centers.
    Write r1 < r2 < r3 < r4 for the target's four matching panels and W for its letter at r1.
        "target"        the shape at r1, r2, r3, r4
        "source"        the shape at r2 and r4 only, with W at r2. All its other letters are new.
        "source_first"  the source with a new letter in place of W (used when nothing is written)
        "source_stop"   the shape at r1 only, holding W (after " W" this list is finished)
    The pair also holds "ask", "panels" (r1 to r4) and "written" (the letter W)."""
    rng = random.Random(seed)
    layouts = list(itertools.combinations(range(6), 4))
    for candidate in itertools.count():
        ask = LIST_ASK[candidate % 3]
        panels = list(layouts[(candidate // 3 + rng.randrange(len(layouts))) % len(layouts)])
        letters = rng.sample(LETTERS, 12)
        target_letters, new_letters = letters[:6], letters[6:]
        W = target_letters[panels[0]]
        source_letters = [W if panel == panels[1] else letter for panel, letter in enumerate(new_letters)]
        stop_letters = [W if panel == panels[0] else letter for panel, letter in enumerate(new_letters)]
        centers = [[rng.randint(106, 278), rng.randint(106, 278)] for _ in range(6)]
        target = list_side(rng, ask, panels, target_letters, centers)
        source = list_side(rng, ask, [panels[1], panels[3]], source_letters, centers)
        source_stop = list_side(rng, ask, [panels[0]], stop_letters, centers)
        yield {"candidate": candidate, "ask": ask, "panels": panels, "written": W, "target": target,
               "source": source, "source_first": {**source, "letters": new_letters}, "source_stop": source_stop}


def _list_pair_passes(model, pair):
    """The no-patch check of a candidate: every next token that the experiments read must be right."""
    W, target = pair["written"], matching_letters(pair["target"])
    checks = [   # side, letters written, end on a letter?, the next token the model must give
        ("target", [], False, " " + target[0]), ("target", [W], False, " " + target[1]), ("target", [W], True, ","),
        ("source", [W], False, " " + matching_letters(pair["source"])[1]), ("source", [W], True, ","),
        ("source_first", [], False, " " + matching_letters(pair["source_first"])[0]),
        ("source_stop", [], False, " " + W), ("source_stop", [W], True, "<end>")]
    return all(run_list(model, pair[side], written, end, record=False).answer == want for side, written, end, want in checks)


def add_heldout_sources(pairs):
    """Add the extra images of the held-out pairs, in place. `pairs` are all held-out pairs, in order.
        "target_repeat"   the target with W also at r3, so W sits in two matching panels
        "source_third"    the shape at r3 and r4, with W at r3
        "source_set"      the shape at r1 and r3, with W at r1
        "source_other"    the shape at r1 (holding W) and at the first later panel that holds another
                          shape in the target. Missing when there is no such panel.
        "source_written"  the shape at r1 and at the panel just before it, which holds W. Its next
                          panel, r1, is already written in the target. Missing when r1 is panel 0."""
    def side(rng, pair, matching, w_at, unused):
        letters = rng.sample(unused, 6)
        letters[w_at] = pair["written"]
        return list_side(rng, pair["ask"], matching, letters, pair["target"]["centers"])

    rng = random.Random(4321)
    for pair in pairs:
        r, target = pair["panels"], pair["target"]
        unused = [c for c in LETTERS if c not in target["letters"]]
        pair["source_third"] = side(rng, pair, [r[2], r[3]], r[2], unused)
        pair["source_set"] = side(rng, pair, [r[0], r[2]], r[0], unused)
        pair["target_repeat"] = {**target, "letters": [pair["written"] if p == r[2] else c for p, c in enumerate(target["letters"])]}
    rng = random.Random(777)
    for pair in pairs:
        r, target = pair["panels"], pair["target"]
        unused = [c for c in LETTERS if c not in target["letters"] + pair["source"]["letters"]]
        later = [p for p in range(6) if p not in r and p > r[0]]
        if later:
            pair["source_other"] = side(rng, pair, [r[0], later[0]], r[0], unused)
        if r[0] > 0:
            pair["source_written"] = side(rng, pair, [r[0] - 1, r[0]], r[0] - 1, unused)


def list_pairs(model, split="heldout", num_pairs=None, seed=LIST_SEED):
    """The paper's list pairs for one split: "search" (200 pairs), "validation" (100) or "heldout" (500).

    Candidates come from `list_candidates`. A candidate is kept if the model gives the right next
    token in all eight no-patch runs of `_list_pair_passes`. The kept pairs are numbered in order
    ("index") and the splits are ranges of that numbering, as in `one_answer_pairs`. The checks are
    saved in dataset/, so they run only once. Held-out pairs also get the images of `add_heldout_sources`."""
    first, last = SPLITS[split]
    if num_pairs is not None:
        last = min(last, first + num_pairs)
    cache_file = cache_path(model, f"list_seed{seed}.json")
    checked = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    pairs = []
    for pair in list_candidates(seed):
        if len(pairs) == last:
            break
        key = str(pair["candidate"])
        if key not in checked:
            checked[key] = _list_pair_passes(model, pair)
            CACHE.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps(checked))
        if checked[key]:
            pair["index"] = len(pairs)                         # the pair's number among the kept pairs
            pairs.append(pair)
    add_heldout_sources(pairs[SPLITS["heldout"][0]:])
    return pairs[first:last]


# ------------------------------------------------------------------------------------ the progress pairs
# Sections 5.1 to 5.3 draw their own pairs. The matching panels are fixed by each experiment, the
# shapes sit at the panel center, and the other panels hold a square, circle, diamond or plus.
PROGRESS_SEEDS = {"progress": 510, "position": 610, "cursor": 710}     # the first seed of each experiment
_FILLERS = ("square", "circle", "diamond", "plus")


def sample_progress_pair(rng):
    """Section 5.1. The source has written one letter, from the target's first matching panel (panel
    1, 2 or 3). The source's next panel is the target's third matching panel. The target has written
    nothing. The two ask for different shapes and share no letter."""
    source_ask, target_ask = rng.sample(LIST_ASK, 2)
    shared = rng.randrange(0, 3)
    target_panels = sorted([shared] + rng.sample(range(shared + 1, 6), 3))
    source_panels = [shared, target_panels[2]]
    letters = rng.sample(LETTERS, 12)
    s, t = letters[:6], letters[6:]
    source = list_side(rng, source_ask, source_panels, s, written=[s[shared]], other_shapes=_FILLERS)
    target = list_side(rng, target_ask, target_panels, t, other_shapes=_FILLERS)
    return {"source": source, "target": target, "shared_panel": shared,
            "no_patch": {"source": s[source_panels[1]], "target": t[shared]},
            # the target's answer after the patch: its own first answer, its second answer (progress),
            # its letter at the source's next panel (the source's pointer), or the source's letter (a copy)
            "letters": {"own": t[target_panels[0]], "second": t[target_panels[1]],
                        "pointer": t[target_panels[2]], "copy": s[source_panels[1]]}}


def sample_position_pair(rng):
    """Section 5.2. The source has its shape at panels 2, 4, 5 and has written panels 2 and 4. The
    target has its shape at panels 1, 2, 3, 6 and has written nothing."""
    source_ask, target_ask = rng.sample(LIST_ASK, 2)
    letters = rng.sample(LETTERS, 12)
    s, t = letters[:6], letters[6:]
    source = list_side(rng, source_ask, [1, 3, 4], s, written=[s[1], s[3]], other_shapes=_FILLERS)
    target = list_side(rng, target_ask, [0, 1, 2, 5], t, other_shapes=_FILLERS)
    return {"source": source, "target": target, "no_patch": {"source": s[4], "target": t[0]},
            # a set of panels predicts the target's own first answer, a count of two its third match,
            # an ordinal position its first match after panel 4
            "letters": {"set": t[0], "one_skipped": t[1], "count": t[2], "panel_4": t[3], "panel_5": t[4],
                        "position": t[5], "copy": s[4]}}


def sample_cursor_pair(rng):
    """Section 5.3. The target has its shape at panels 1, 3, 4, 6 and has written panels 1 and 3. The
    source images are the target's image with two letters swapped: in "source" the last written
    letter sits in panel 4, in "source_first" the first written letter sits in panel 4."""
    ask = rng.choice(LIST_ASK)
    target = list_side(rng, ask, [0, 2, 3, 5], [""] * 6, other_shapes=_FILLERS)
    t = rng.sample(LETTERS, 6)
    target.update(letters=t, written=[t[0], t[2]])
    return {"target": target, "no_patch": {"target": t[3]},
            "source": {**target, "letters": [t[0], t[1], t[3], t[2], t[4], t[5]]},
            "source_first": {**target, "letters": [t[3], t[1], t[2], t[0], t[4], t[5]]},
            "letters": {"own": t[3], "after_panel_4": t[5], "repeat_last": t[2], "repeat_first": t[0]}}


def progress_pairs(model, experiment, num_pairs=500, pairs_per_seed=100):
    """The pairs of one progress experiment: "progress" (5.1), "position" (5.2) or "cursor" (5.3).

    Pairs are drawn from the experiment's first seed until 100 pass the no-patch check, then from the
    next seed, and so on. A pair passes when the unpatched runs listed in pair["no_patch"] give the
    expected next letter. The checks are saved in dataset/. The paper uses 500 pairs (five seeds)."""
    sample = {"progress": sample_progress_pair, "position": sample_position_pair, "cursor": sample_cursor_pair}[experiment]
    cache_file = cache_path(model, f"progress_{experiment}.json")
    checked = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    pairs = []
    for seed in itertools.count(PROGRESS_SEEDS[experiment]):
        rng, kept = random.Random(seed), 0
        for draw in itertools.count():
            if kept == pairs_per_seed or len(pairs) == num_pairs:
                break
            pair, key = sample(rng), f"{seed}:{draw}"
            if key not in checked:
                checked[key] = all(run_list(model, pair[side], record=False).answer.strip() == letter
                                   for side, letter in pair["no_patch"].items())
                CACHE.mkdir(exist_ok=True)
                cache_file.write_text(json.dumps(checked))
            if checked[key]:
                pair["index"] = len(pairs)
                pairs.append(pair)
                kept += 1
        if len(pairs) == num_pairs:
            return pairs


# ------------------------------------------------------------------------------------ stopping
def stop_directions(model, pair):
    """The two cases of the stopping experiment for one list pair (Section 5.4).

    Both runs end on the written letter W, with no comma after it, so the next token is the model's
    choice between a comma (the list goes on) and the end of its turn (the list stops). The target
    of the pair has matching panels left and goes on. The pair's "source_stop" image has no other
    matching panel and stops. Yields (name, source, target) for each direction of the patch, or
    nothing when one of the two runs does not make its own choice with no patch."""
    going = run_list(model, pair["target"], [pair["written"]], end_on_letter=True)
    finished = run_list(model, pair["source_stop"], [pair["written"]], end_on_letter=True)
    if going.answer == "," and finished.answer == "<end>":
        yield "finished_list_goes_on", going, finished
        yield "continuing_list_stops", finished, going


# ------------------------------------------------------------------------------------ lists on COCO photographs
# Section 5.4 and Appendix G ask for a list on natural images. This needs a local copy of COCO
# val2014 in the standard layout (nothing is downloaded):
#     <coco>/annotations/instances_val2014.json
#     <coco>/val2014/COCO_val2014_000000xxxxxx.jpg
COCO_SEED = 20260925
COCO_TAIL = " Answer with the names only, separated by a comma and a space."
COCO_QUESTIONS = {"animal": "What animals are in this image?" + COCO_TAIL,
                  "vehicle": "What vehicles are in this image?" + COCO_TAIL,
                  "furniture": "What furniture is in this image?" + COCO_TAIL,
                  "object": "What objects are in this image?" + COCO_TAIL}
# The COCO categories of each question, and how many images of each the paper takes. The object
# question accepts all 80 categories.
COCO_GROUPS = {"animal": ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"),
               "vehicle": ("bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"),
               "furniture": ("chair", "couch", "bed", "dining table")}
COCO_IMAGES = {"animal": 400, "vehicle": 300, "furniture": 300, "object": 400}
# The words the model may write for each COCO category. Names with spaces are separated by "|".
COCO_WORDS = {
    "person": "person people man men woman women boy boys girl girls child children kid kids player players skier skiers surfer surfers skateboarder rider riders baby lady guy",
    "bicycle": "bicycle bike", "car": "car taxi van suv", "motorcycle": "motorcycle motorbike scooter moped", "airplane": "airplane plane jet aircraft airliner",
    "bus": "bus buses", "train": "train locomotive", "truck": "truck lorry pickup", "boat": "boat ship sailboat canoe kayak yacht",
    "traffic light": "traffic light|traffic lights|stoplight", "fire hydrant": "fire hydrant|hydrant", "stop sign": "stop sign", "parking meter": "parking meter",
    "bench": "bench benches", "bird": "bird duck seagull pigeon goose geese swan parrot owl eagle", "cat": "cat kitten", "dog": "dog puppy puppies",
    "horse": "horse pony ponies", "sheep": "sheep lamb", "cow": "cow cattle bull calf ox", "elephant": "elephant", "bear": "bear", "zebra": "zebra",
    "giraffe": "giraffe", "backpack": "backpack", "umbrella": "umbrella", "handbag": "handbag purse bag", "tie": "tie necktie", "suitcase": "suitcase luggage",
    "frisbee": "frisbee", "skis": "ski skis", "snowboard": "snowboard", "sports ball": "ball|soccer ball|tennis ball|football|basketball|baseball", "kite": "kite",
    "baseball bat": "bat|baseball bat", "baseball glove": "glove|baseball glove|mitt", "skateboard": "skateboard", "surfboard": "surfboard|surf board",
    "tennis racket": "racket|tennis racket|racquet", "bottle": "bottle", "wine glass": "wine glass|glass", "cup": "cup mug", "fork": "fork", "knife": "knife knives",
    "spoon": "spoon", "bowl": "bowl", "banana": "banana", "apple": "apple", "sandwich": "sandwich", "orange": "orange", "broccoli": "broccoli", "carrot": "carrot",
    "hot dog": "hot dog|hotdog", "pizza": "pizza", "donut": "donut doughnut", "cake": "cake", "chair": "chair", "couch": "couch sofa",
    "potted plant": "plant|potted plant|flower pot|houseplant", "bed": "bed", "dining table": "table|dining table", "toilet": "toilet",
    "tv": "tv|television|monitor|screen", "laptop": "laptop|computer", "mouse": "mouse|computer mouse", "remote": "remote|remote control",
    "keyboard": "keyboard", "cell phone": "phone|cell phone|cellphone|smartphone|mobile phone", "microwave": "microwave", "oven": "oven|stove",
    "toaster": "toaster", "sink": "sink", "refrigerator": "refrigerator|fridge", "book": "book", "clock": "clock", "vase": "vase", "scissors": "scissors",
    "teddy bear": "teddy bear|teddy|stuffed animal", "hair drier": "hair dryer|hairdryer|hair drier", "toothbrush": "toothbrush"}


def coco_words(category):
    """Every word accepted for one category: the listed names, and their plurals with an added s."""
    names = COCO_WORDS[category].split("|") if "|" in COCO_WORDS[category] else COCO_WORDS[category].split()
    return names + [name + "s" for name in names if not name.endswith("s") and " " not in name]


COCO_CATEGORY_OF_WORD = {word: category for category in COCO_WORDS for word in coco_words(category)}


def coco_list_images(coco, part="test"):
    """The COCO val2014 images of the paper's list questions, in the paper's order.

    A category counts as present when its annotated area is at least 1% of the image. An image is
    taken for the animal, vehicle, or furniture question when at least two categories of that
    group are present, and for the object question when at least three categories are present.
    The images of each question are shuffled with a fixed seed. The first 4 of the first three
    questions are set aside (`part="set_aside"`; the language experiment takes its averages there),
    and the next 400, 300, 300, and 400 are the test images (`part="test"`, 1,304 images).
    Each entry holds "id", "file", "group", "width", "height", and "boxes": the boxes
    [x, y, width, height] of every annotated category of the question's group."""
    data = json.loads((Path(coco) / "annotations" / "instances_val2014.json").read_text())
    names = {category["id"]: category["name"] for category in data["categories"]}
    images = {image["id"]: image for image in data["images"]}
    boxes, area = {}, {}
    for annotation in data["annotations"]:
        if annotation.get("iscrowd"):
            continue
        key, name = annotation["image_id"], names[annotation["category_id"]]
        boxes.setdefault(key, {}).setdefault(name, []).append(annotation["bbox"])
        area[key, name] = area.get((key, name), 0) + annotation["area"]
    found = {group: [] for group in COCO_QUESTIONS}
    for key, by_name in boxes.items():
        image = images[key]
        present = [name for name in by_name if area[key, name] / (image["width"] * image["height"]) >= 0.01]
        for group in found:
            in_group = [name for name in by_name if group == "object" or name in COCO_GROUPS[group]]
            if sum(name in in_group for name in present) >= (3 if group == "object" else 2):
                found[group].append({"id": key, "file": image["file_name"], "group": group, "width": image["width"],
                                     "height": image["height"], "boxes": {name: by_name[name] for name in in_group}})
    rng = random.Random(COCO_SEED)
    chosen = []
    for group, entries in found.items():
        entries.sort(key=lambda entry: entry["id"])
        rng.shuffle(entries)
        aside = 0 if group == "object" else 4
        chosen += entries[:aside] if part == "set_aside" else entries[aside:aside + COCO_IMAGES[group]]
    return chosen


def coco_image(coco, entry):
    """Load one COCO image, resized down to multiples of 32 pixels so that each image token is a
    32 x 32 cell. Returns (image, regions): `regions` maps each category to its boxes
    [x0, y0, x1, y1] in the pixels of the resized image."""
    image = Image.open(Path(coco) / "val2014" / entry["file"]).convert("RGB")
    width, height = entry["width"] // 32 * 32, entry["height"] // 32 * 32
    sx, sy = width / entry["width"], height / entry["height"]
    regions = {name: [[x * sx, y * sy, (x + w) * sx, (y + h) * sy] for x, y, w, h in group_boxes]
               for name, group_boxes in entry["boxes"].items()}
    return image.resize((width, height)), regions


def run_coco_list(model, coco, entry):
    """Let the model write its list for one COCO image, then run it again after its first item.

    The list is valid when its first two items name two different annotated categories, A then
    B, and the prefilled answer "Answer: A," gives the first token of B again. Returns
    (run, case) for a valid list and (None, reason) otherwise. The run has the named tokens
    "colon", "first_item" (all tokens of A) and "last_comma". `case` holds the two categories
    ("first", "second"), the first token of each written item ("first_token", "second_token"),
    and "region_positions": the image tokens inside the boxes of each category."""
    import torch
    from utils.model import panel_of_image_tokens
    from utils.patching import run_model
    from utils.patching import write_tokens
    image, regions = coco_image(coco, entry)
    question = COCO_QUESTIONS[entry["group"]]
    written = write_tokens(model, run_model(model, image, question, record=False), max_tokens=20).strip()
    items = [item.strip().lower().rstrip(".") for item in written.split(",") if item.strip()]
    if len(items) < 2:
        return None, "fewer than two items"
    first, second = (COCO_CATEGORY_OF_WORD.get(item) for item in items[:2])
    if first is None or second is None or first not in regions or second not in regions or first == second:
        return None, "the first two items are not two different annotated categories"
    run = run_model(model, image, question, prefill=f"Answer: {items[0]},")
    tokenizer = model.processor.tokenizer
    first_token, second_token = (tokenizer.decode(tokenizer.encode(" " + item, add_special_tokens=False)[:1]) for item in items[:2])
    if run.answer != second_token:
        return None, "the prefilled answer changes the next token"
    inside = lambda boxes: torch.as_tensor(panel_of_image_tokens(run.info, boxes) >= 0, device=run.device)
    region_positions = {name: run.image_positions[inside(boxes)] for name, boxes in regions.items()}
    if len(region_positions[first]) == 0 or len(region_positions[second]) == 0:
        return None, "a region is smaller than one image token"
    colon = run.find(":")
    run.name_token("colon", colon).name_token("first_item", list(range(colon + 1, run.last))).name_token("last_comma", run.last)
    return run, {"first": first, "second": second, "first_token": first_token, "second_token": second_token,
                 "written": written, "region_positions": region_positions}


# ------------------------------------------------------------------------------------ the colored-shapes task
# Section 6.2. Four panels, one filled shape in each, no letters. Exactly one shape is cyan, and the
# question asks for its shape, in English, German, French, or Chinese. The answer is a word that
# differs between the languages, so we can tell the concept (which shape) from the language.
COLOR_SHAPES = ("circle", "square", "star", "heart")
LANGUAGES = ("en", "de", "fr", "zh")
CYAN = "#00C8E8"
OTHER_COLORS = {"orange": "#FF7F00", "green": "#76D300", "pink": "#FF3EA5", "yellow": "#FFD500",
                "red": "#E8202A", "purple": "#8A3FFC"}
COLOR_QUESTIONS = {
    "en": "What shape is the cyan marker? Reply with one shape word only.",
    "de": "Welche Form hat die cyanfarbene Markierung? Antworte nur mit einem Wort für die Form.",
    "fr": "Quelle est la forme du marqueur cyan ? Réponds avec un seul mot de forme.",
    "zh": "青色标记是什么形状？只用一个形状词回答。",
}
# The words we accept for each shape in each language. The first one is the usual answer.
SHAPE_WORDS = {
    "en": {"circle": ["circle"], "square": ["square"], "star": ["star"], "heart": ["heart"]},
    "de": {"circle": ["kreis"], "square": ["quadrat"], "star": ["stern", "sterne"], "heart": ["herz"]},
    "fr": {"circle": ["cercle"], "square": ["carré", "carre"], "star": ["étoile", "etoile"], "heart": ["cœur", "coeur"]},
    "zh": {"circle": ["圆形", "圆"], "square": ["正方形", "方形"], "star": ["星形", "星", "五角星"], "heart": ["心形", "心"]},
}
COLOR_SEED = 20260918
COLOR_SPLITS = {"search": (12, 0), "heldout": (500, 10_000)}       # number of images, seed offset


def draw_color_strip(shapes, colors, centers=None):
    """A row of blank panels with one filled, colored shape in each and no letters. Returns a PIL image.

    shapes    one of COLOR_SHAPES per panel;   colors   one fill color per panel (name or hex code)
    centers   optional (x, y) of each shape inside its panel; default is the panel center"""
    image = Image.new("RGB", (len(shapes) * PANEL, PANEL), "white")
    draw = ImageDraw.Draw(image)
    for i, (shape, color) in enumerate(zip(shapes, colors)):
        x0, r = i * PANEL, 72
        draw.rectangle((x0, 0, x0 + PANEL - 1, PANEL - 1), outline="#666666", width=3)
        cx, cy = (PANEL // 2, PANEL // 2) if centers is None else centers[i]
        cx, style = x0 + cx, {"fill": {"cyan": CYAN, **OTHER_COLORS}.get(color, color), "outline": "black", "width": 1}
        if shape == "circle":
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), **style)
        elif shape == "square":
            draw.polygon([(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)], **style)
        elif shape == "star":
            draw.polygon(_star(cx, cy, int(r * 1.3)), **style)
        elif shape == "heart":
            draw.polygon(_heart(cx, cy - r // 8, int(r * 1.35)), **style)
        else:
            raise ValueError(f"unknown shape {shape!r}; choose from {COLOR_SHAPES}")
    image.regions = [[i * PANEL, 0, (i + 1) * PANEL, PANEL] for i in range(len(shapes))]
    image.shapes = list(shapes)
    return image


def sample_color_image(index, split="heldout", seed=COLOR_SEED):
    """Image number `index` of one split. The cyan shape and its panel cycle with the index, so
    every shape is asked equally often at every panel. Returns a dict with "index", "shape" (the
    cyan shape, which is the answer), "panel", "shapes", "colors", "centers"."""
    rng = random.Random(seed + COLOR_SPLITS[split][1] + index)
    shape, panel = COLOR_SHAPES[index % 4], (index + index // 4) % 4
    centers = [(rng.randint(130, 254), rng.randint(130, 254)) for _ in range(4)]
    others = [s for s in COLOR_SHAPES if s != shape]
    rng.shuffle(others)
    colors = rng.sample(list(OTHER_COLORS.values()), 3)
    shapes = [shape if i == panel else others.pop() for i in range(4)]
    colors = [CYAN if i == panel else colors.pop() for i in range(4)]
    return {"index": index, "shape": shape, "panel": panel, "shapes": shapes, "colors": colors, "centers": centers}


def color_image(spec):
    return draw_color_strip(spec["shapes"], spec["colors"], spec["centers"])


def color_images(split="heldout", num_images=None):
    """The paper's images for one split: "search" (12 images) or "heldout" (500 images).

    The model answers every one of these questions correctly in all four languages
    (language/main_experiments/01), so no image is left out and no model is needed here.
    Each image also gets its three sources: the next three images, which show another cyan shape."""
    assert split in COLOR_SPLITS, f"the colored-shapes task has the splits {list(COLOR_SPLITS)}"
    total = COLOR_SPLITS[split][0]
    images = []
    for index in range(total if num_images is None else min(num_images, total)):
        spec = sample_color_image(index, split)
        spec["sources"] = [sample_color_image((index + step) % total, split) for step in (1, 2, 3)]
        images.append(spec)
    return images


def run_color_question(model, spec, language, record=True):
    """Run the model on one colored-shapes image in one language. Nothing is prefilled, so the
    last prompt token is the line break after "assistant". It is named "last"."""
    from utils.patching import run_model
    return run_model(model, color_image(spec), COLOR_QUESTIONS[language], prefill="", record=record)


def color_cases(model, spec, target_languages=("de", "fr", "zh")):
    """The source-target cases of one target image: each of its three source images, with every
    pair of different languages. The paper's targets ask in German, French, or Chinese, and the
    source asks in any other language, which gives 3 x 9 = 27 cases per target image.

    Yields (source_run, target_run, case). `case` names the two languages and the two shapes."""
    targets = {language: run_color_question(model, spec, language, record=False) for language in target_languages}
    for source_spec in spec["sources"]:
        sources = {language: run_color_question(model, source_spec, language) for language in LANGUAGES}
        for target_language in target_languages:
            for source_language in LANGUAGES:
                if source_language != target_language:
                    yield sources[source_language], targets[target_language], {
                        "source_image": source_spec["index"], "source_language": source_language,
                        "target_language": target_language, "source_shape": source_spec["shape"], "target_shape": spec["shape"]}


def _find_word(text, word):
    """Where `word` starts in `text`, or -1. Latin words must stand alone; Chinese words may be inside others."""
    if re.search(r"[一-鿿]", word):
        return text.find(word)
    match = re.search(r"(?<![a-zäöüéèêàçœ])" + re.escape(word) + r"(?![a-zäöüéèêàçœ])", text)
    return match.start() if match else -1


def first_word(text, candidates):
    """`candidates` maps a name to a list of words. Returns the name whose word comes first in the
    text, or "other" if none is there. An answer of several tokens is scored as text this way."""
    text, best, where = text.lower(), "other", len(text) + 1
    for name, words in candidates.items():
        for word in words:
            start = _find_word(text, word)
            if 0 <= start < where:
                best, where = name, start
    return best


def answer_language(text, shape):
    """The language in which the text names `shape`: "en", "de", "fr", "zh", or "other"."""
    return first_word(text, {language: SHAPE_WORDS[language][shape] for language in LANGUAGES})


def patched_answer(text, case):
    """What a patched target wrote, when the source asks in another language about another shape:
        "language"   the target's shape in the source's language (the language moved)
        "unchanged"  the target's shape in the target's language
        "word"       the source's shape in the source's language (the source's word was copied)
        "concept"    the source's shape in the target's language (the concept moved)
        "other"      none of these
    `case` comes from `color_cases`."""
    source, target = SHAPE_WORDS[case["source_language"]], SHAPE_WORDS[case["target_language"]]
    return first_word(text, {"language": source[case["target_shape"]], "unchanged": target[case["target_shape"]],
                             "word": source[case["source_shape"]], "concept": target[case["source_shape"]]})


def shape_word_token(model, language, shape):
    """The token id that starts the word for `shape`, as the head search of the paper scored it:
    the word written after a space (German capitalized), or the bare word in Chinese."""
    word = SHAPE_WORDS[language][shape][0]
    text = word if language == "zh" else " " + (word.capitalize() if language == "de" else word)
    return model.processor.tokenizer.encode(text, add_special_tokens=False)[0]


# ------------------------------------------------------------------------------------ COCO in German
# The COCO list questions (see `coco_list_images`) asked in German, to test the language heads on
# photographs. The first written token tells the language of the answer.
COCO_TAIL_GERMAN = " Antworte nur mit den Namen, getrennt durch ein Komma und ein Leerzeichen."
COCO_QUESTIONS_GERMAN = {"animal": "Welche Tiere sind auf diesem Bild?" + COCO_TAIL_GERMAN,
                         "vehicle": "Welche Fahrzeuge sind auf diesem Bild?" + COCO_TAIL_GERMAN,
                         "furniture": "Welche Möbel sind auf diesem Bild?" + COCO_TAIL_GERMAN}
COCO_GERMAN_WORDS = {
    "bird": "vogel vögel ente enten möwe taube gans schwan", "cat": "katze katzen", "dog": "hund hunde", "horse": "pferd pferde",
    "sheep": "schaf schafe lamm", "cow": "kuh kühe rind rinder stier", "elephant": "elefant elefanten", "bear": "bär bären",
    "zebra": "zebra zebras", "giraffe": "giraffe giraffen", "bicycle": "fahrrad fahrräder", "car": "auto autos wagen pkw",
    "motorcycle": "motorrad motorräder roller", "airplane": "flugzeug flugzeuge", "bus": "bus busse", "train": "zug züge lok",
    "truck": "lkw lastwagen laster", "boat": "boot boote schiff schiffe", "chair": "stuhl stühle", "couch": "sofa couch",
    "bed": "bett betten", "dining table": "tisch tische esstisch"}


def coco_language(token, categories):
    """Does a first answer token start an English or a German name of one of these COCO categories?
    Returns "english", "german", "both" (the names start the same, as in Giraffe), or "other"."""
    token = token.strip().lower()
    english = bool(token) and any(word.startswith(token) for c in categories for word in coco_words(c))
    german = bool(token) and any(word.startswith(token) for c in categories for word in COCO_GERMAN_WORDS.get(c, "").split())
    return "both" if english and german else "english" if english else "german" if german else "other"


def coco_argument(default="data/coco"):
    """Read `--coco FOLDER` from the command line (or the COCO_FOLDER variable) and remove it, so
    that the usual arguments can be parsed afterwards. The folder must hold COCO val2014 (see
    `coco_list_images`). Returns the folder, or None with a message if it is not there."""
    folder = os.environ.get("COCO_FOLDER", default)
    if "--coco" in sys.argv:
        at = sys.argv.index("--coco")
        folder = sys.argv[at + 1]
        del sys.argv[at:at + 2]
    if not os.path.exists(os.path.join(folder, "annotations", "instances_val2014.json")):
        print(f"COCO val2014 was not found in {folder!r}. Pass --coco FOLDER. Nothing was run.")
        return None
    return folder


# ==================================================================================================
# Appendix A and B
# ==================================================================================================
# ------------------------------------------------------------------------------------ how reliable is the task
ALL_LETTERS = tuple(string.ascii_uppercase)


def sample_one_answer_pair_all_letters(candidate, seed=ONE_ANSWER_SEED):
    """A pair of the one-answer task with its eight letters drawn from all 26 capital letters.

    Shapes, order and centers are those of `sample_one_answer_pair`. The letters are drawn again
    from their own seed, as in the paper's first reliability check (Appendix A.7)."""
    pair = sample_one_answer_pair(candidate, seed)
    letters = random.Random(seed + 30_000 + candidate).sample(ALL_LETTERS, 8)
    pair["source"]["letters"], pair["target"]["letters"] = letters[:4], letters[4:]
    return pair


# ------------------------------------------------------------------------------------ other requests
# Appendix B.5. The four-panel task with another kind of request. The question names a color, an
# object, a color and a shape, or the paper's shape with a three-letter word in place of the letter.
#     color     circles filled with distinct colors          "inside the red circle"    request word: the color
#     category  a drawn object with a letter to its right    "next to the cat"          request word: the object
#     two       colored shapes; the asked item has a same-color and a same-shape distractor
#                                                            "inside the red star"      request words: both
#     words     the paper's shapes, each with a word; the model writes the word's initial
#     wordq     the same images; the question asks for the word, and the model writes it
OTHER_REQUESTS = ("color", "category", "two", "words", "wordq")
OTHER_REQUEST_QUESTIONS = {
    "color": "Answer with the letter only. What is the letter inside the {item} circle?",
    "category": "Answer with the letter only. What is the letter next to the {item}?",
    "two": "Answer with the letter only. What is the letter inside the {item}?",
    "words": "Answer with the letter only. What is the letter inside the {item}?",
    "wordq": "Answer with the word only. What is the word inside the {item}?",
}
REQUEST_COLORS = {"red": "#E8202A", "blue": "#2050E0", "green": "#2AA52A", "yellow": "#FFD500", "orange": "#FF7F00",
                  "purple": "#8A3FFC"}
CATEGORY_OBJECTS = {"cat": "\U0001F431", "dog": "\U0001F436", "car": "\U0001F697", "key": "\U0001F511", "bell": "\U0001F514",
           "horse": "\U0001F434", "cow": "\U0001F42E", "pig": "\U0001F437", "frog": "\U0001F438", "bee": "\U0001F41D",
           "egg": "\U0001F95A", "apple": "\U0001F34E", "fish": "\U0001F41F", "bird": "\U0001F426", "sun": "\U0001F31E",
           "book": "\U0001F4D5", "tree": "\U0001F333", "house": "\U0001F3E0"}
# Three-letter words that are one token in upper case, lower case and capitalized.
LABEL_WORDS = ("CAT SUN CAR BUS PEN KEY BOX MAP FOX ANT FIG ICE LEG NET PIE VAN BAT BED BIN CAN CAP DAM DOT EAR FIN GAS "
               "HIP LAB LOG MAT PAD PAN POT RAM RAT ROW SEA TOP WEB").split()
TWO_SHAPES = ("star", "circle", "square", "triangle", "heart")
OTHER_REQUEST_SEED = 150
# The category images need a color emoji font. Set EMOJI_FONT to its path if it is elsewhere.
EMOJI_FONT = os.environ.get("EMOJI_FONT", "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf")


def _emoji_font():
    """The color emoji font, or a plain message that says how to get it."""
    if not os.path.exists(EMOJI_FONT):
        raise SystemExit(f"The color emoji font was not found at '{EMOJI_FONT}'. Install Noto Color Emoji "
                         "(on Ubuntu: apt install fonts-noto-color-emoji) or set EMOJI_FONT to its path.")
    return ImageFont.truetype(EMOJI_FONT, 109)             # a bitmap emoji font loads at this size only


def _filled_shape(draw, shape, cx, cy, fill, width):
    """One shape of the list task's size, with a fill color. Returns the vertical shift of its letter."""
    r, shape_dy, letter_dy = _SIZES["large"][shape]
    style = {"fill": fill, "outline": "black", "width": width}
    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), **style)
    elif shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), **style)
    elif shape == "triangle":
        draw.polygon(_polygon(cx, cy + shape_dy, r, 3), **style)
    elif shape == "diamond":
        draw.polygon(_polygon(cx, cy, r, 4), **style)
    elif shape == "star":
        draw.polygon(_star(cx, cy + shape_dy, r), **style)
    elif shape == "plus":
        draw.polygon(_plus(cx, cy, r, r * 0.42), **style)
    else:
        draw.polygon(_heart(cx, cy, r), **style)
    return letter_dy


_CATEGORY_EMOJI = {}


def _category_emoji(name, size=96):
    """The object drawn as a color emoji, 96 pixels wide."""
    if name not in _CATEGORY_EMOJI:
        font = _emoji_font()
        tile = Image.new("RGBA", (160, 160), (255, 255, 255, 0))
        ImageDraw.Draw(tile).text((80, 80), CATEGORY_OBJECTS[name], font=font, embedded_color=True, anchor="mm")
        _CATEGORY_EMOJI[name] = tile.crop(tile.getbbox()).resize((size, size), Image.Resampling.LANCZOS)
    return _CATEGORY_EMOJI[name]


def other_request_image(variant, side, centers):
    """Draw one side of an other-request pair. `side` holds "shapes" (the item of each panel, as its
    name in the question) and "letters" (the label of each panel: a letter, or a word)."""
    image = Image.new("RGB", (4 * PANEL, PANEL), "white")
    draw = ImageDraw.Draw(image)
    font, small = ImageFont.truetype(str(FONT), 48), ImageFont.truetype(str(FONT), 32)
    for i, (item, label, (x, cy)) in enumerate(zip(side["shapes"], side["letters"], centers)):
        draw.rectangle((i * PANEL, 0, (i + 1) * PANEL - 1, PANEL - 1), outline="#666666", width=3)
        cx = i * PANEL + x
        if variant == "color":
            _filled_shape(draw, "circle", cx, cy, REQUEST_COLORS[item], 4)
            draw.text((cx, cy + 1), label, font=font, fill="black", anchor="mm", stroke_width=2, stroke_fill="white")
        elif variant == "category":
            icon = _category_emoji(item)
            image.paste(icon, (cx - 58 - 48, cy - 48), icon)
            draw.text((cx + 58, cy), label, font=font, fill="black", anchor="mm")
        elif variant == "two":
            color, shape = item.split()
            dy = _filled_shape(draw, shape, cx, cy, REQUEST_COLORS[color], 4)
            draw.text((cx, cy + dy), label, font=font, fill="black", anchor="mm", stroke_width=2, stroke_fill="white")
        else:                                                   # words, wordq
            dy = _filled_shape(draw, item, cx, cy, "white", 6)
            draw.text((cx, cy + dy), label, font=small, fill="black", anchor="mm", stroke_width=3, stroke_fill="white")
    image.regions = [[i * PANEL, 0, (i + 1) * PANEL, PANEL] for i in range(4)]
    image.shapes, image.letters = list(side["shapes"]), list(side["letters"])
    return image


def sample_other_request_pair(variant, rng):
    """One source-target pair of one variant, drawn from `rng`. Both sides hold the same four items
    in different orders and ask for different items, placed so that the outcome letters T, D, K, P
    and R are all different (as in `sample_one_answer_pair`). Returns {"source", "target", "centers"};
    each side has "shapes" (item names), "letters" (labels) and "ask"."""
    name = (lambda item: f"{item[0]} {item[1]}") if variant == "two" else (lambda item: item)
    if variant in ("color", "category"):
        items = rng.sample(list(REQUEST_COLORS if variant == "color" else CATEGORY_OBJECTS), 4)
        source_ask, target_ask = rng.sample(items, 2)
    elif variant == "two":
        c1, c2 = rng.sample(list(REQUEST_COLORS), 2)
        s1, s2 = rng.sample(TWO_SHAPES, 2)
        items = [(c1, s1), (c1, s2), (c2, s1), (c2, s2)]
        target_ask, source_ask = (c1, s1), (c2, s2)
    else:
        items = rng.sample(SHAPES, 4)
        source_ask, target_ask = rng.sample(items, 2)
    source_order = rng.sample(items, 4)
    free = [i for i in range(4) if i not in (source_order.index(source_ask), source_order.index(target_ask))]
    rng.shuffle(free)
    target_order = [None] * 4
    target_order[free[0]], target_order[free[1]] = source_ask, target_ask
    others = [item for item in items if item not in (source_ask, target_ask)]
    rng.shuffle(others)
    for i in [i for i, item in enumerate(target_order) if item is None]:
        target_order[i] = others.pop()
    if variant == "words":                                     # scored on the initial, so eight different initials
        pool = list(LABEL_WORDS)
        rng.shuffle(pool)
        labels = []
        for word in pool:
            if word[0] not in {w[0] for w in labels}:
                labels.append(word)
        labels = labels[:8]
    else:
        labels = rng.sample(LABEL_WORDS if variant == "wordq" else LETTERS, 8)
    centers = [[rng.randint(120, 264), rng.randint(120, 264)] for _ in range(4)]
    return {"centers": centers,
            "source": {"shapes": [name(i) for i in source_order], "letters": labels[:4], "ask": name(source_ask)},
            "target": {"shapes": [name(i) for i in target_order], "letters": labels[4:], "ask": name(target_ask)}}


def is_label(answer, label, variant):
    """Is the next token this label? A letter is " X". Under the "words" question the model writes
    the word's initial, so that is scored. Under "wordq" the word counts in any of three casings."""
    if variant == "words":
        return answer == " " + label[0]
    if variant == "wordq":
        return answer in (" " + label, " " + label.lower(), " " + label.capitalize())
    return answer == " " + label


def run_other_request(model, variant, side, centers, record=True):
    """Run one side and name its tokens: "colon", "shape" (the last request word) and "request"
    (the positions of all request words: one, or the color and the shape word in "two")."""
    from utils.patching import run_model
    image = other_request_image(variant, side, centers)
    run = run_model(model, image, OTHER_REQUEST_QUESTIONS[variant].format(item=side["ask"]), record=record)
    words = side["ask"].split()
    request = [run.find(" " + word) for word in words]
    run.name_token("request", request).name_token("shape", request[-1]).name_token("colon", run.last)
    return run


def other_request_pairs(model, variant, num_pairs=500, seed=OTHER_REQUEST_SEED):
    """The paper's pairs of one variant, and the no-patch answers of every pair tried.

    Pairs are drawn one after another from one random generator. A pair is kept if the model
    answers both of its questions correctly with no patch. Each kept pair has "index", the number
    of the draw, which also seeds its random heads. The checks are saved in dataset/. Returns
    (kept pairs, checks) where checks maps every draw up to the last kept pair to [source right,
    target right]."""
    cache_file = cache_path(model, f"other_requests_{variant}_seed{seed}.json")
    checked = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    rng, pairs, tried = random.Random(seed), [], {}
    for draw in range(2 * num_pairs + 20):
        if len(pairs) == num_pairs:
            break
        pair = sample_other_request_pair(variant, rng)
        if str(draw) not in checked:
            checked[str(draw)] = [is_label(run_other_request(model, variant, pair[side], pair["centers"], record=False).answer,
                                           pair[side]["letters"][pair[side]["shapes"].index(pair[side]["ask"])], variant)
                                  for side in ("source", "target")]
            CACHE.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps(checked))
        tried[draw] = checked[str(draw)]
        if all(checked[str(draw)]):
            pair["index"] = draw
            pairs.append(pair)
    return pairs, tried


def other_request_directions(model, variant, pair):
    """Run both sides of a pair, then use each as the source of the other. Yields (source_run,
    target_run, letters) twice. `letters` are the outcome letters of `outcome_letters`."""
    a, b = (run_other_request(model, variant, pair[side], pair["centers"]) for side in ("source", "target"))
    yield a, b, outcome_letters(pair["source"], pair["target"])
    yield b, a, outcome_letters(pair["target"], pair["source"])


# ==================================================================================================
# Appendix D
# ==================================================================================================
# ------------------------------------------------------------------------------------ Appendix D pairs
# The controls of Appendix D.2 draw their own pairs, as the progress pairs of Sections 5.1 to 5.3 do.
CONTROL_SEED = 810                      # the first of five seeds, 100 kept pairs each
SUBSPACE_SEED = 9404                    # Appendix D.5, one seed for all pairs
CURSOR_IMAGE_SEEDS = {"directions": 8001, "test": 8002}    # Appendix D.5, images with nothing patched


def sample_control_pair(rng):
    """Appendix D.2, the controls of the skip experiment. As in `sample_progress_pair`, the source has
    written its letter from the target's first matching panel (panel 1, 2 or 3), and the target has
    written nothing. Here the source has its shape at that panel and at two random later panels. The
    two ask for different shapes and share no letter."""
    source_ask, target_ask = rng.sample(LIST_ASK, 2)
    shared = rng.randrange(0, 3)
    target_panels = sorted([shared] + rng.sample(range(shared + 1, 6), 3))
    source_panels = sorted([shared] + rng.sample(range(shared + 1, 6), 2))
    letters = rng.sample(LETTERS, 12)
    s, t = letters[:6], letters[6:]
    source = list_side(rng, source_ask, source_panels, s, written=[s[shared]], other_shapes=_FILLERS)
    target = list_side(rng, target_ask, target_panels, t, other_shapes=_FILLERS)
    return {"source": source, "target": target, "shared_panel": shared,
            "no_patch": {"source": s[source_panels[1]], "target": t[shared]},
            # the first name that matches wins, so "pointer" counts only when it is not the target's first two
            "letters": {"own": t[target_panels[0]], "second": t[target_panels[1]], "copy": s[source_panels[1]],
                        "pointer": t[source_panels[1]]}}


def sample_subspace_pair(rng, design):
    """Appendix D.5. The pair of Section 5.2 (`sample_position_pair`): the source has its shape at
    panels 2, 4 and 5, the target at panels 1, 2, 3 and 6. In design "p4" the source has written
    panels 2 and 4, as in Section 5.2. In design "p2" it has written panel 2 only, so its next letter
    is at panel 4 and the target's first match after panel 2 is panel 3."""
    pair = sample_position_pair(rng)
    if design == "p2":
        s, t = pair["source"]["letters"], pair["target"]["letters"]
        pair["source"] = {**pair["source"], "written": [s[1]]}
        pair["no_patch"] = {"source": s[3], "target": t[0]}
        pair["letters"] = {"set": t[0], "count": t[1], "position": t[2], "panel_6": t[5], "copy": s[3]}
    return pair


def _kept_pairs(model, sample, cache_name, seeds, pairs_per_seed, num_pairs, max_tries=None, after_check=None):
    """Draw pairs from each seed in turn until `pairs_per_seed` pass the no-patch check, as in
    `progress_pairs`. `after_check(rng, pair)` runs on each kept pair right after its check, for
    draws that the paper took from the same random generator."""
    cache_file = cache_path(model, cache_name)
    checked = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    pairs = []
    for seed in seeds:
        rng, kept = random.Random(seed), 0
        for draw in itertools.count():
            if kept == pairs_per_seed or len(pairs) == num_pairs or (max_tries and draw == max_tries):
                break
            pair, key = sample(rng), f"{seed}:{draw}"
            if key not in checked:
                checked[key] = all(run_list(model, pair[side], record=False).answer.strip() == letter
                                   for side, letter in pair["no_patch"].items())
                CACHE.mkdir(exist_ok=True)
                cache_file.write_text(json.dumps(checked))
            if checked[key]:
                if after_check:
                    after_check(rng, pair)
                pair["index"] = len(pairs)
                pairs.append(pair)
                kept += 1
        if len(pairs) == num_pairs:
            break
    return pairs


def control_pairs(model, num_pairs=500):
    """The pairs of the Appendix D.2 controls: 100 kept pairs from each of the seeds 810 to 814. A
    pair is kept when the source's next letter is its second match and the target's is its first.
    Each kept pair also gets "absent": a letter drawn in neither image, prefilled in one control."""
    def add_absent_letter(rng, pair):
        used = set(pair["source"]["letters"]) | set(pair["target"]["letters"])
        pair["absent"] = rng.choice([c for c in LETTERS if c not in used])
    seeds = range(CONTROL_SEED, CONTROL_SEED + (num_pairs + 99) // 100)
    return _kept_pairs(model, sample_control_pair, "progress_controls.json", seeds, 100, num_pairs,
                       after_check=add_absent_letter)


def subspace_pairs(model, design, num_pairs=500):
    """The pairs of Appendix D.5, design "p4" or "p2", all from seed 9404 (at most 2 x num_pairs draws)."""
    return _kept_pairs(model, lambda rng: sample_subspace_pair(rng, design), f"progress_subspace_{design}.json",
                       [SUBSPACE_SEED], num_pairs, num_pairs, max_tries=2 * num_pairs)


def sample_cursor_image(rng):
    """Appendix D.5. One list image with the asked shape at four random panels, the shapes at the
    panel centers, and six new letters. Nothing is written."""
    ask = rng.choice(LIST_ASK)
    matching = sorted(rng.sample(range(6), 4))
    letters = rng.sample(LETTERS, 6)
    return list_side(rng, ask, matching, letters, other_shapes=_FILLERS)


def cursor_images(part, num_images):
    """The images of Appendix D.5: "directions" (120 images in the paper) or "test" (500 images)."""
    rng = random.Random(CURSOR_IMAGE_SEEDS[part])
    return [{**sample_cursor_image(rng), "index": i} for i in range(num_images)]


# ==================================================================================================
# Appendix E
# ==================================================================================================
# ------------------------------------------------------------------------------------ longer strips (Appendix E.2)
# The model makes few errors on six panels, so Appendix E.2 lets it write whole lists on strips of
# ten panels with seven matching panels. The other panels hold a square, circle, diamond or plus.
LONG_STRIP_SEED = 19
_LONG_FILLERS = ("square", "circle", "diamond", "plus")


def sample_long_strip(rng, n_panels, n_matching, index):
    """One strip of the list task with `n_panels` panels, `n_matching` of which hold the asked shape.
    Returns a side dict, as `list_side` does, so `run_list` can run it."""
    ask = LIST_ASK[index % 3]
    matching = sorted(rng.sample(range(n_panels), n_matching))
    fillers = [s for s in _LONG_FILLERS if s not in CONFUSED[ask]]
    shapes = [ask if p in matching else rng.choice(fillers) for p in range(n_panels)]
    letters = rng.sample(LETTERS, n_panels)
    centers = [[rng.randint(106, 278), rng.randint(106, 278)] for _ in range(n_panels)]
    return {"index": index, "shapes": shapes, "letters": letters, "ask": ask, "matching": matching,
            "centers": centers, "written": []}


def long_strips(num_images=400, n_panels=10, n_matching=7, seed=LONG_STRIP_SEED):
    """The strips of Appendix E.2, drawn in order from one seed. No model check is needed."""
    rng = random.Random(seed)
    return [sample_long_strip(rng, n_panels, n_matching, index) for index in range(num_images)]


def last_written_panel(side, written):
    """The panel of the last written letter (the cursor). If that letter is not in the image, the
    rightmost written panel. -1 when nothing in the image is written."""
    if written and written[-1] in side["letters"]:
        return side["letters"].index(written[-1])
    return max([side["letters"].index(c) for c in written if c in side["letters"]], default=-1)


def next_matching_panel(side, written):
    """The first matching panel after the cursor whose letter is not yet written, or None."""
    cursor = last_written_panel(side, written)
    left = [p for p in side["matching"] if p > cursor and side["letters"][p] not in written]
    return left[0] if left else None


def label_list_answer(side, written, letter):
    """Name one answer of a whole list:
        correct   the next matching panel          skip     a later unwritten matching panel
        repeat    a letter already written         back     an unwritten matching panel before the cursor
        filler    a panel without the asked shape  extra    a letter after every matching panel is done
        foreign   a letter that is not in the image"""
    target = next_matching_panel(side, written)
    if letter in written:
        return "repeat"
    if letter not in side["letters"]:
        return "foreign"
    panel = side["letters"].index(letter)
    if target is None:
        return "extra"
    if panel == target:
        return "correct"
    if panel in side["matching"]:
        return "skip" if panel > target else "back"
    return "filler"


# ==================================================================================================
# Appendix F
# ==================================================================================================
# ------------------------------------------------------------------------------------ the object task
# Each object: its emoji, then the words we accept in English, German, French, and Chinese. Every
# object has a different word in each language. The other panels draw their objects from all 24;
# the asked object is one of the 16 of OBJECT_POOL.
OBJECTS = {
    "apple": ("\U0001F34E", ["apple"], ["apfel"], ["pomme"], ["苹果"]),
    "dog": ("\U0001F436", ["dog", "puppy"], ["hund"], ["chien"], ["狗", "小狗"]),
    "cat": ("\U0001F431", ["cat", "kitten"], ["katze"], ["chat"], ["猫", "小猫"]),
    "car": ("\U0001F697", ["car"], ["auto", "wagen"], ["voiture"], ["汽车", "车", "小汽车"]),
    "house": ("\U0001F3E0", ["house", "home"], ["haus"], ["maison"], ["房子", "房屋", "屋"]),
    "tree": ("\U0001F333", ["tree"], ["baum"], ["arbre"], ["树", "树木"]),
    "fish": ("\U0001F41F", ["fish"], ["fisch"], ["poisson"], ["鱼"]),
    "bird": ("\U0001F426", ["bird"], ["vogel"], ["oiseau"], ["鸟", "小鸟"]),
    "sun": ("\U0001F31E", ["sun"], ["sonne"], ["soleil"], ["太阳"]),
    "moon": ("\U0001F319", ["moon"], ["mond"], ["lune"], ["月亮", "月"]),
    "book": ("\U0001F4D5", ["book"], ["buch"], ["livre"], ["书", "书本"]),
    "key": ("\U0001F511", ["key"], ["schlüssel", "schluessel"], ["clé", "clef", "cle"], ["钥匙"]),
    "bell": ("\U0001F514", ["bell"], ["glocke"], ["cloche"], ["铃", "铃铛", "钟"]),
    "umbrella": ("☔", ["umbrella"], ["regenschirm", "schirm"], ["parapluie"], ["雨伞", "伞"]),
    "horse": ("\U0001F434", ["horse"], ["pferd"], ["cheval"], ["马"]),
    "cow": ("\U0001F42E", ["cow"], ["kuh"], ["vache"], ["牛", "奶牛"]),
    "pig": ("\U0001F437", ["pig"], ["schwein"], ["cochon", "porc"], ["猪"]),
    "frog": ("\U0001F438", ["frog"], ["frosch"], ["grenouille"], ["青蛙", "蛙"]),
    "bee": ("\U0001F41D", ["bee"], ["biene"], ["abeille"], ["蜜蜂", "蜂"]),
    "egg": ("\U0001F95A", ["egg"], ["ei"], ["œuf", "oeuf"], ["鸡蛋", "蛋"]),
    "cheese": ("\U0001F9C0", ["cheese"], ["käse", "kaese"], ["fromage"], ["奶酪", "芝士", "乳酪"]),
    "bread": ("\U0001F35E", ["bread"], ["brot"], ["pain"], ["面包"]),
    "shoe": ("\U0001F45F", ["shoe", "sneaker"], ["schuh", "turnschuh"], ["chaussure", "basket"], ["鞋", "鞋子", "运动鞋"]),
    "hat": ("\U0001F3A9", ["hat"], ["hut", "zylinder"], ["chapeau"], ["帽子", "礼帽"]),
}
# The 16 objects the model names correctly at all four panels in all four languages (21 of the 24
# pass; the paper keeps 16), in the order the images cycle through them.
OBJECT_POOL = ("key", "bell", "umbrella", "horse", "cow", "pig", "frog", "bee", "egg", "cheese",
               "apple", "dog", "cat", "car", "house", "tree")
OBJECT_WORDS = {language: {name: OBJECTS[name][1 + i] for name in OBJECTS} for i, language in enumerate(LANGUAGES)}
OBJECT_QUESTIONS = {
    "en": "What is inside the circle? Answer with one word only.",
    "de": "Was ist im Kreis? Antworte nur mit einem Wort.",
    "fr": "Qu’y a-t-il dans le cercle ? Réponds avec un seul mot.",
    "zh": "圆圈里面是什么？只用一个词回答。",
}
OUTLINES = ("square", "diamond", "triangle", "heart")     # the shapes around the other three objects
OBJECT_SEED = 20260918
OBJECT_SPLITS = {"search": (12, 0), "heldout": (100, 10_000)}       # number of images, seed offset
# The objects are drawn with the Noto Color Emoji font, which is not shipped with this repository.
# On Ubuntu it is in the package fonts-noto-color-emoji. Give another path with EMOJI_FONT=...


def _emoji(char, size):
    font = _emoji_font()
    tile = Image.new("RGBA", (160, 160), (255, 255, 255, 0))
    ImageDraw.Draw(tile).text((80, 80), char, font=font, embedded_color=True, anchor="mm")
    return tile.crop(tile.getbbox()).resize((size, size), Image.Resampling.LANCZOS)


def draw_object_strip(objects, outlines, centers):
    """A row of blank panels, one outlined shape with one drawn object inside in each. Returns a PIL image.
    objects   one name of OBJECTS per panel;   outlines   "circle" or one of OUTLINES per panel"""
    image = Image.new("RGB", (len(objects) * PANEL, PANEL), "white")
    draw = ImageDraw.Draw(image)
    for i, (name, outline, (cx, cy)) in enumerate(zip(objects, outlines, centers)):
        x0, r = i * PANEL, 92
        draw.rectangle((x0, 0, x0 + PANEL - 1, PANEL - 1), outline="#666666", width=3)
        cx, style = x0 + cx, {"fill": "white", "outline": "black", "width": 6}
        if outline == "circle":
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), **style)
        elif outline == "square":
            draw.polygon([(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)], **style)
        elif outline == "diamond":
            draw.polygon(_polygon(cx, cy, int(r * 1.25), 4), **style)
        elif outline == "triangle":
            draw.polygon(_polygon(cx, cy + r // 5, int(r * 1.55), 3), **style)
        else:
            draw.polygon(_heart(cx, cy - r // 8, int(r * 1.35)), **style)
        icon = _emoji(OBJECTS[name][0], 96)
        image.paste(icon, (cx - 48, cy - 48 + (14 if outline == "triangle" else 0)), icon)
    image.regions = [[i * PANEL, 0, (i + 1) * PANEL, PANEL] for i in range(len(objects))]
    return image


def sample_object_image(index, split="heldout", seed=OBJECT_SEED):
    """Image number `index` of one split of the object task. The asked object (inside the circle)
    and its panel cycle with the index. Returns a dict with "index", "object", "panel", "objects",
    "outlines", "centers"."""
    rng = random.Random(seed + OBJECT_SPLITS[split][1] + index)
    name, panel = OBJECT_POOL[index % len(OBJECT_POOL)], (index + index // len(OBJECT_POOL)) % 4
    centers = [(rng.randint(130, 254), rng.randint(130, 254)) for _ in range(4)]
    others = rng.sample([o for o in OBJECTS if o != name], 3)
    outlines = rng.sample(OUTLINES, 3)
    shapes = ["circle" if i == panel else outlines.pop() for i in range(4)]
    objects = [name if i == panel else others.pop() for i in range(4)]
    return {"index": index, "object": name, "panel": panel, "objects": objects, "outlines": shapes, "centers": centers}


def object_images(split="heldout", num_images=None):
    """The images of one split of the object task: "search" (12 images) or "heldout" (100 images).
    No image is left out. Each image gets its three sources: the next three images."""
    total = OBJECT_SPLITS[split][0]
    images = []
    for index in range(total if num_images is None else min(num_images, total)):
        spec = sample_object_image(index, split)
        spec["sources"] = [sample_object_image((index + step) % total, split) for step in (1, 2, 3)]
        images.append(spec)
    return images


def run_object_question(model, spec, language, record=True):
    """Run the model on one object image in one language, with nothing prefilled (see `run_color_question`)."""
    from utils.patching import run_model
    image = draw_object_strip(spec["objects"], spec["outlines"], spec["centers"])
    return run_model(model, image, OBJECT_QUESTIONS[language], prefill="", record=record)


# ------------------------------------------------------------------------------------ new nouns
# Appendix F.6 adds the direction of a new English noun to a run and checks in which language the
# model writes it. These are the nouns, with the words accepted in German, French, and Chinese.
NEW_NOUNS = {
    "apple": (["apfel"], ["pomme"], ["苹果"]), "dog": (["hund"], ["chien"], ["狗"]),
    "house": (["haus"], ["maison"], ["房子", "房屋", "住宅"]), "car": (["auto", "wagen"], ["voiture"], ["汽车", "车"]),
    "water": (["wasser"], ["eau"], ["水"]), "book": (["buch"], ["livre"], ["书", "书籍"]),
    "tree": (["baum"], ["arbre"], ["树", "树木"]), "sun": (["sonne"], ["soleil"], ["太阳"]),
    "moon": (["mond"], ["lune"], ["月亮", "月球"]), "milk": (["milch"], ["lait"], ["牛奶", "奶"]),
    "bread": (["brot"], ["pain"], ["面包"]), "fish": (["fisch"], ["poisson"], ["鱼"]),
    "bird": (["vogel"], ["oiseau"], ["鸟"]), "flower": (["blume", "blüte"], ["fleur"], ["花", "花朵"]),
    "chair": (["stuhl"], ["chaise"], ["椅子"]), "door": (["tür", "tuer"], ["porte"], ["门"]),
    "cat": (["katze"], ["chat"], ["猫"]), "mountain": (["berg"], ["montagne"], ["山"]),
    "king": (["könig", "koenig"], ["roi"], ["国王"]), "bed": (["bett"], ["lit"], ["床"]),
}


def new_nouns(task):
    """The nouns added in a task: the second half of NEW_NOUNS, after leaving out the objects shown
    in the object task's images (10 nouns for colored shapes, 7 for objects)."""
    names = [n for n in NEW_NOUNS if task.name == "colors" or n not in OBJECT_POOL]
    return names[len(names) // 2:]


def noun_language(text, noun):
    """The language the text names `noun` in: "de", "fr", "zh", "en", or "other"."""
    for language, words in zip(("de", "fr", "zh"), NEW_NOUNS[noun]):
        if first_word(text, {language: words}) == language:
            return language
    return "en" if first_word(text, {"en": [noun]}) == "en" else "other"


# ------------------------------------------------------------------------------------ one interface for both tasks
class LanguageTask:
    """The colored-shapes task or the object task. A concept is a shape or an object name.

        task.images(split, n)          the image specs of a split, each with its three "sources"
        task.concept(spec)             the asked concept of an image
        task.run(model, spec, lang)    a recorded run of the question in one language
        task.cases(model, spec)        the source-target cases of one target image (see `color_cases`)
        task.words                     words[language][concept] = the accepted words
        task.word_token(model, lang, concept)   the token whose logit the head searches score"""

    def __init__(self, name):
        self.name = name
        colors = name == "colors"
        self.words = SHAPE_WORDS if colors else OBJECT_WORDS
        self._images = color_images if colors else object_images
        self._run = run_color_question if colors else run_object_question
        self._key = "shape" if colors else "object"

    def images(self, split="heldout", num_images=None):
        return self._images(split, num_images)

    def concept(self, spec):
        return spec[self._key]

    def run(self, model, spec, language, record=True):
        return self._run(model, spec, language, record)

    def cases(self, model, spec, target_languages=("de", "fr", "zh"), same_language=False):
        """Yields (source_run, target_run, case) for each of the three sources and each pair of
        languages: different languages, or with same_language=True the same language only."""
        targets = {language: self.run(model, spec, language) for language in target_languages}
        for source_spec in spec["sources"]:
            sources = {language: self.run(model, source_spec, language) for language in LANGUAGES}
            for target_language in target_languages:
                for source_language in LANGUAGES:
                    if (source_language == target_language) == same_language:
                        yield sources[source_language], targets[target_language], {
                            "source_image": source_spec["index"], "source_language": source_language,
                            "target_language": target_language, "source_concept": self.concept(source_spec),
                            "target_concept": self.concept(spec), "source_panel": source_spec["panel"]}

    def answer_language(self, text, concept):
        """The language in which the text names `concept`, or "other"."""
        return first_word(text, {language: self.words[language][concept] for language in LANGUAGES})

    def patched_answer(self, text, case):
        """As `patched_answer` of utils/dataset.py: "language", "unchanged", "word", "concept" or "other"."""
        source, target = self.words[case["source_language"]], self.words[case["target_language"]]
        return first_word(text, {"language": source[case["target_concept"]], "unchanged": target[case["target_concept"]],
                                 "word": source[case["source_concept"]], "concept": target[case["source_concept"]]})

    def same_language_answer(self, text, case):
        """For a source in the target's own language: "unchanged", "concept" (the source's concept), or "other"."""
        words = self.words[case["target_language"]]
        return first_word(text, {"unchanged": words[case["target_concept"]], "concept": words[case["source_concept"]]})

    def names_concept(self, text, concept):
        """Does the text name this concept in any of the four languages?"""
        return self.answer_language(text, concept) != "other"

    def item_at(self, spec, panel):
        """The concept that an image shows at one panel."""
        return spec["shapes" if self.name == "colors" else "objects"][panel]

    def word_token(self, model, language, concept):
        """The token the head searches score. Colored shapes: the start of the word after a space, as
        the paper's 500-image search scored it (`shape_word_token`). Objects: the first token the
        model itself writes for this object in this language (`first_answer_tokens`)."""
        if self.name == "colors":
            return shape_word_token(model, language, concept)
        return first_answer_tokens(model, self)[language][concept]


def language_task(name="colors"):
    assert name in ("colors", "objects"), "the language tasks are 'colors' and 'objects'"
    return LanguageTask(name)


def task_argument(default="colors"):
    """Read `--task colors|objects` from the command line and remove it, so that the usual arguments
    can be parsed afterwards (as `coco_argument`). Returns the LanguageTask."""
    name = default
    if "--task" in sys.argv:
        at = sys.argv.index("--task")
        name = sys.argv[at + 1]
        del sys.argv[at:at + 2]
    return language_task(name)


def first_answer_tokens(model, task):
    """For each language and concept, the most common first token of the model's own correct answers
    with no patch: on the search images first, then on the held-out images for a concept the search
    images do not show. Saved in dataset/, so the runs happen once. Returns {language: {concept: id}}."""
    from utils.patching import write_tokens
    cache_file = cache_path(model, f"first_answer_tokens_{task.name}.json")
    if cache_file.exists():
        return {language: {c: int(t) for c, t in table.items()} for language, table in json.loads(cache_file.read_text()).items()}
    tokens = {}
    for split in ("search", "heldout"):
        counts = collections.defaultdict(collections.Counter)
        for spec in task.images(split):
            concept = task.concept(spec)
            if all(concept in tokens.get(language, {}) for language in LANGUAGES):
                continue
            for language in LANGUAGES:
                run = task.run(model, spec, language, record=False)
                if task.answer_language(write_tokens(model, run), concept) == language:
                    counts[(language, concept)][run.answer_id] += 1
        for (language, concept), counter in counts.items():
            tokens.setdefault(language, {}).setdefault(concept, counter.most_common(1)[0][0])
    CACHE.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(tokens))
    return tokens


def average_head_outputs(model, task, heads, languages=LANGUAGES):
    """The average output of each head at the last prompt token over the 12 search images of a task,
    for each language of the question. Returns {language: {(layer, head): vector}}."""
    images, averages = task.images("search"), {}
    for language in languages:
        total = dict.fromkeys(heads, 0)
        for spec in images:                                 # one run at a time, so few runs stay in memory
            run = task.run(model, spec, language)
            total = {(l, h): total[(l, h)] + run.head_output(l, h, "last") for l, h in heads}
        averages[language] = {head: value / len(images) for head, value in total.items()}
    return averages


# ==================================================================================================
# Appendix G: other model sizes and families
# ==================================================================================================
def pairs_of_8b(kind, split="heldout", num_pairs=None, device="cuda:0"):
    """The one-answer or list pairs of the 8B, for runs on another model size.

    The paper's runs on the 2B, 4B and 32B use the 8B's pairs: the same images, questions and
    splits, kept by the 8B's no-patch check. Each experiment then scores every one of these pairs
    on the other size, so that size's own mistakes count. This loads the 8B, builds the pairs (the
    checks are saved in dataset/, so they run only once), and frees the 8B again.

    kind   "one_answer" (one_answer_pairs) or "list" (list_pairs)."""
    import gc

    import torch

    from utils.model import MODEL_ID, load_model
    make = {"one_answer": one_answer_pairs, "list": list_pairs}[kind]
    model = load_model(MODEL_ID, device)
    pairs = make(model, split, num_pairs)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return pairs


def progress_pairs_of(model, model_id, experiment, num_pairs=500):
    """`progress_pairs` for any model. These pairs are kept by the model's own no-patch check, as
    in the paper's runs. Each model keeps its checks in its own folder of dataset/, so the checks of
    one model are never read for another."""
    return progress_pairs(model, experiment, num_pairs)


def _says(answer, letter):
    """Does an answer token name this letter? Case-folded: Idefics3 writes lists in lower case."""
    return answer.strip().upper() == letter


def family_one_answer_pairs(vlm, num_pairs, first=0):
    """The one-answer pairs for a model of another family (Appendix G.2). Candidates are the paper's
    (`sample_one_answer_pair`), from candidate `first` on. A pair is kept when the model answers both
    of its questions right with no patch. `vlm` is a `utils.patching.OtherVLM`."""
    pairs, candidate = [], first
    while len(pairs) < num_pairs:
        pair = sample_one_answer_pair(candidate)
        if all(_says(vlm.run(vlm.prompt(one_answer_image(side), one_answer_question(side["ask"])))[0],
                     side["letters"][side["shapes"].index(side["ask"])]) for side in (pair["source"], pair["target"])):
            pairs.append({**pair, "index": candidate})
        candidate += 1
    return pairs


def family_progress_pairs(vlm, experiment, num_pairs):
    """The pairs of the skip ("progress") or cursor ("position") experiment for a model of another
    family. They are drawn as in `progress_pairs`, from the experiment's first seed, and kept when
    the no-patch runs in pair["no_patch"] give the expected next letter."""
    sample = {"progress": sample_progress_pair, "position": sample_position_pair}[experiment]
    rng, pairs, draw = random.Random(PROGRESS_SEEDS[experiment]), [], 0
    while len(pairs) < num_pairs:
        pair = sample(rng)
        if all(_says(vlm.run(vlm.prompt(list_image(pair[side]), list_question(pair[side]["ask"]), pair[side]["written"]))[0], letter)
               for side, letter in pair["no_patch"].items()):
            pairs.append({**pair, "index": draw})
        draw += 1
    return pairs


def sample_list_question(rng, matches=None):
    """One plain list question (Appendix G.2): six panels, the shape we ask for in 2 to 4 of them
    (or `matches` of them), the other panels filled as in the progress pairs. Returns a list side
    (see `list_side`) whose "expected" is the list the model should write."""
    ask = rng.choice(LIST_ASK)
    panels = sorted(rng.sample(range(6), matches or rng.choice([2, 3, 4])))
    letters = rng.sample(LETTERS, 6)
    side = list_side(rng, ask, panels, letters, other_shapes=("square", "circle", "diamond", "plus"))
    side["expected"] = [letters[panel] for panel in panels]
    return side


# ==================================================================================================
# Appendix G: other layouts, prompts, and natural images
# ==================================================================================================
# ------------------------------------------------------------------------------------ wordings and layouts
WORDINGS = {
    "paper": (ONE_ANSWER_PROMPT, LIST_PROMPT),
    "one_sentence_after": ("Which letter is inside the {shape}? Reply with one letter only.",
                           "Answer with the letters only, separated by a comma and a space. What are the letters inside every {shape}?"),
    "two_sentences_after": ("Look at the {shape}. What letter is written in it? Answer with the letter only.",
                            "Answer with the letters only, separated by a comma and a space. List the letter inside each {shape}."),
    "right_to_left": (ONE_ANSWER_PROMPT, LIST_PROMPT + " List them from right to left."),
}
LAYOUTS = ("row", "grid", "column", "mirrored_row", "canvas", "photo", "separate", "text")
PLACE_MARGIN = 24                 # a place on a canvas is the box around a shape, this many pixels wider than the shape
CANVAS_RADIUS = 100               # the largest shape radius, used to place shapes on a canvas
CANVAS_DISTANCE = 250             # the smallest distance between two shape centers on a canvas
ROW_HEIGHTS = (60, 160, 260)      # row heights of the reading-order rule (see `reading_order`)
PHOTOS = 250                      # the photo layout uses the first 250 images of COCO val2014, in file-name order


def make_variant(layout="row", wording="paper", coco=None):
    assert layout in LAYOUTS and wording in WORDINGS, (layout, wording)
    return {"layout": layout, "wording": wording, "coco": coco, "name": f"{layout}_{wording}"}


def variant_argument(layout="row", wording="paper"):
    """Read --layout, --wording and --coco from the command line and remove them, so that the usual
    arguments can be parsed afterwards (as `coco_argument` does). Returns a variant."""
    values = {"--layout": layout, "--wording": wording, "--coco": os.environ.get("COCO_FOLDER")}
    for flag in values:
        if flag in sys.argv:
            at = sys.argv.index(flag)
            values[flag] = sys.argv[at + 1]
            del sys.argv[at:at + 2]
    if values["--layout"] == "photo" and not values["--coco"]:
        sys.exit("the photo layout needs --coco FOLDER with COCO val2014 (the images in FOLDER/val2014/)")
    return make_variant(values["--layout"], values["--wording"], values["--coco"])


def folder_argument(flag, what):
    """Read a required folder `flag` (for example --comics) from the command line and remove it."""
    if flag not in sys.argv:
        sys.exit(f"{flag} FOLDER is required: {what}")
    at = sys.argv.index(flag)
    folder = sys.argv[at + 1]
    del sys.argv[at:at + 2]
    return folder


def panel_origin(index, count, layout):
    """Top-left corner of panel `index` of `count` panels."""
    if layout == "grid":
        columns = 2 if count == 4 else 3
        return (index % columns) * PANEL, (index // columns) * PANEL
    if layout == "column":
        return 0, index * PANEL
    if layout == "mirrored_row":
        return (count - 1 - index) * PANEL, 0
    return index * PANEL, 0


def place_box(center, radius=CANVAS_RADIUS):
    """The place of a shape on a canvas: the box around it with a margin."""
    r = radius + PLACE_MARGIN
    return [center[0] - r, center[1] - r, center[0] + r, center[1] + r]


def draw_panels(shapes, letters, centers=None, size="small", layout="row"):
    """Panels in one of the layouts row, grid, column, mirrored_row. Like `draw_strip`, the image
    carries `regions` (panel i is region i, wherever it is drawn), `shapes` and `letters`."""
    count = len(shapes)
    columns = {"grid": 2 if count == 4 else 3, "column": 1}.get(layout, count)
    rows = math.ceil(count / columns)
    font = ImageFont.truetype(str(FONT), 48)
    image = Image.new("RGB", (columns * PANEL, rows * PANEL), "white")
    draw = ImageDraw.Draw(image)
    image.regions = []
    for i, (shape, letter) in enumerate(zip(shapes, letters)):
        x0, y0 = panel_origin(i, count, layout)
        draw.rectangle((x0, y0, x0 + PANEL - 1, y0 + PANEL - 1), outline="#666666", width=3)
        cx, cy = (PANEL // 2, PANEL // 2) if centers is None else centers[i]
        draw_shape(draw, shape, letter, x0 + cx, y0 + cy, font, size)
        image.regions.append([x0, y0, x0 + PANEL, y0 + PANEL])
    image.shapes, image.letters = list(shapes), list(letters)
    return image


def draw_canvas(shapes, letters, centers, background=None):
    """Shapes at absolute `centers` on one canvas with no panels: 768 x 768 for four shapes, 960 x 960
    for six. `background` is an optional photograph drawn under them. The shapes are the large ones
    of the list task, with a white fill and a black outline. Place i is the box around shape i."""
    side = 768 if len(shapes) <= 4 else 960
    image = Image.new("RGB", (side, side), "white") if background is None else background.convert("RGB").resize((side, side))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT), 48)
    for shape, letter, (cx, cy) in zip(shapes, letters, centers):
        draw_shape(draw, shape, letter, cx, cy, font, size="large")
    image.regions = [place_box(center) for center in centers]
    image.shapes, image.letters = list(shapes), list(letters)
    return image


def sample_centers(rng, count, side, radius=CANVAS_RADIUS, distance=CANVAS_DISTANCE, x_gap=64):
    """Random shape centers on a square canvas. Shapes never touch (centers at least `distance`
    apart), and any two centers differ by at least `x_gap` pixels in x."""
    low, high = radius + PLACE_MARGIN, side - radius - PLACE_MARGIN
    for _ in range(100_000):
        points = [(rng.randint(low, high), rng.randint(low, high)) for _ in range(count)]
        if all(math.dist(a, b) >= distance and abs(a[0] - b[0]) >= x_gap for i, a in enumerate(points) for b in points[:i]):
            return [list(p) for p in points]
    raise RuntimeError("could not place the shapes")


def reading_order(centers, places, row_height=160):
    """The order in which the model lists shapes on a canvas: take the topmost remaining shape and
    every shape within `row_height` pixels below it as a row, and name the leftmost shape of that row."""
    left, order = list(places), []
    while left:
        top = min(left, key=lambda p: centers[p][1])
        row = [p for p in left if centers[p][1] <= centers[top][1] + row_height]
        order.append(min(row, key=lambda p: centers[p][0]))
        left.remove(order[-1])
    return order


def clear_order(centers, places):
    """The reading order of `places` if it is the same for every row height of ROW_HEIGHTS, else None."""
    orders = {tuple(reading_order(centers, places, height)) for height in ROW_HEIGHTS}
    return list(orders.pop()) if len(orders) == 1 else None


def sample_ordered_centers(rng, count, side, subsets):
    """Canvas centers numbered in reading order, on a layout where the order of all shapes, and of
    every group of two or more places inside each of `subsets`, does not depend on the row height."""
    for _ in range(100_000):
        centers = sample_centers(rng, count, side)
        order = clear_order(centers, range(count))
        if order is None:
            continue
        centers = [centers[p] for p in order]
        groups = [list(g) for s in subsets for k in range(2, len(s) + 1) for g in itertools.combinations(s, k)]
        if all(clear_order(centers, g) == g for g in groups):
            return centers
    raise RuntimeError("no layout with a clear reading order")


_PHOTO_FILES = {}


def photo(coco, index):
    """Background number `index` of the photo layout (one of the first 250 COCO val2014 images)."""
    if coco not in _PHOTO_FILES:
        _PHOTO_FILES[coco] = sorted(os.listdir(Path(coco) / "val2014"))[:PHOTOS]
    return Image.open(Path(coco) / "val2014" / _PHOTO_FILES[coco][index % PHOTOS])


def variant_image(side, variant, size):
    """Draw one side of a pair in the variant's layout. `size` is "small" (one answer) or "large" (list)."""
    layout = variant["layout"]
    if layout in ("canvas", "photo"):
        background = photo(variant["coco"], side["photo"]) if layout == "photo" else None
        return draw_canvas(side["shapes"], side["letters"], side["canvas_centers"], background)
    return draw_panels(side["shapes"], side["letters"], side.get("centers"), size, layout)


# ------------------------------------------------------------------------------------ one answer on other images
def sample_variant_one_answer_pair(candidate, variant, seed=ONE_ANSWER_SEED):
    """The pair of `sample_one_answer_pair`. On a canvas or a photo, the shapes get absolute centers
    instead, which the source and the target share (as they share the panels)."""
    pair = sample_one_answer_pair(candidate, seed)
    if variant["layout"] in ("canvas", "photo"):
        centers = sample_centers(random.Random(seed + 50_000 + candidate), 4, 768)
        for side in (pair["source"], pair["target"]):
            side.update(canvas_centers=centers, photo=candidate)
    return pair


def variant_one_answer_question(shape, variant):
    return WORDINGS[variant["wording"]][0].format(shape=shape)


def run_variant_one_answer(model, side, variant, record=True):
    """Run one side of a one-answer pair in the variant and name the "shape" and "colon" tokens."""
    from utils.patching import run_model
    run = run_model(model, variant_image(side, variant, "small"), variant_one_answer_question(side["ask"], variant), record=record)
    run.name_token("shape", run.find(" " + side["ask"]))
    run.name_token("colon", run.last)
    return run


def variant_one_answer_pairs(model, variant, split="heldout", num_pairs=None, seed=ONE_ANSWER_SEED):
    """The pairs of `one_answer_pairs`, drawn and asked in the variant. A candidate is kept if the
    model answers both of its questions correctly in the variant with no patch. The checks are saved
    in dataset/, one file per variant."""
    first, last = SPLITS[split]
    if num_pairs is not None:
        last = min(last, first + num_pairs)
    cache_file = cache_path(model, f"one_answer_{variant['name']}_seed{seed}.json")
    checked = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    pairs, candidate = [], 0
    while len(pairs) < last:
        pair = sample_variant_one_answer_pair(candidate, variant, seed)
        if str(candidate) not in checked:
            checked[str(candidate)] = all(
                run_variant_one_answer(model, side, variant, record=False).answer == " " + side["letters"][side["shapes"].index(side["ask"])]
                for side in (pair["source"], pair["target"]))
            CACHE.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps(checked))
        if checked[str(candidate)]:
            pair["index"] = len(pairs)
            pairs.append(pair)
        candidate += 1
    return pairs[first:last]


def variant_both_directions(model, pair, variant):
    """`both_directions` in the variant: yields (source_run, target_run, outcome letters) twice."""
    a, b = run_variant_one_answer(model, pair["source"], variant), run_variant_one_answer(model, pair["target"], variant)
    yield a, b, outcome_letters(pair["source"], pair["target"])
    yield b, a, outcome_letters(pair["target"], pair["source"])


# ------------------------------------------------------------------------------------ lists on other images
def variant_list_question(shape, variant):
    return WORDINGS[variant["wording"]][1].format(shape=shape)


def panel_descriptions(side):
    """The text layout: the six panels described in words, as in "Panel 1: a star with the letter G."."""
    return "Six panels from left to right. " + " ".join(
        f"Panel {i + 1}: a {shape} with the letter {letter}." for i, (shape, letter) in enumerate(zip(side["shapes"], side["letters"])))


def separate_images(side):
    """The separate layout: each panel of a list side as its own 384 x 384 image."""
    centers = side.get("centers") or [None] * len(side["shapes"])
    return [draw_panels([shape], [letter], None if center is None else [center], "large")
            for shape, letter, center in zip(side["shapes"], side["letters"], centers)]


def name_list_tokens(run, side, written, end_on_letter):
    """Name the tokens of a list run as `run_list` does: "shape", "colon", "answers", "commas",
    "last_answer", "last_comma"."""
    colon = run.last - 2 * len(written) + (1 if written and end_on_letter else 0)
    assert run.decode(run.ids[colon]) == ":", "the answer letters must be one token each"
    answers = [colon + 1 + 2 * i for i in range(len(written))]
    commas = [position + 1 for position in answers if position + 1 <= run.last]
    run.name_token("shape", run.find(" " + side["ask"]))
    run.name_token("colon", colon).name_token("answers", answers).name_token("commas", commas)
    if answers:
        run.name_token("last_answer", answers[-1])
    if commas:
        run.name_token("last_comma", commas[-1])
    return run


def run_list_variant(model, side, variant, written=None, end_on_letter=False, record=True):
    """`run_list` in the variant: the same named tokens, with the variant's image and prompt. In the
    text layout the "image tokens" of the run are the tokens of the panel descriptions."""
    from utils.patching import run_model_images, run_model_text
    from utils.patching import run_model
    written = side["written"] if written is None else list(written)
    question, prefill = variant_list_question(side["ask"], variant), list_prefill(written, end_on_letter)
    if variant["layout"] == "text":
        run = run_model_text(model, panel_descriptions(side), question, prefill, record=record)
    elif variant["layout"] == "separate":
        run = run_model_images(model, separate_images(side), question, prefill, record=record)
    else:
        run = run_model(model, variant_image(side, variant, "large"), question, prefill, record=record)
    return name_list_tokens(run, side, written, end_on_letter)


def variant_list_candidates(variant, seed=LIST_SEED):
    """The candidates of `list_candidates`. On a canvas, every image of a pair gets the same absolute
    centers, numbered in reading order, so that panel i is the i-th shape the model should list."""
    for pair in list_candidates(seed):
        if variant["layout"] == "canvas":
            centers = sample_ordered_centers(random.Random(seed + 50_000 + pair["candidate"]), 6, 960, [pair["panels"]])
            for name in ("target", "source", "source_first", "source_stop"):
                pair[name] = {**pair[name], "canvas_centers": centers}
        yield pair


def _variant_list_passes(model, pair, variant):
    """The no-patch check of `_list_pair_passes`, in the variant."""
    W, target = pair["written"], matching_letters(pair["target"])
    checks = [("target", [], False, " " + target[0]), ("target", [W], False, " " + target[1]), ("target", [W], True, ","),
              ("source", [W], False, " " + matching_letters(pair["source"])[1]), ("source", [W], True, ","),
              ("source_first", [], False, " " + matching_letters(pair["source_first"])[0]),
              ("source_stop", [], False, " " + W), ("source_stop", [W], True, "<end>")]
    return all(run_list_variant(model, pair[side], variant, written, end, record=False).answer == want
               for side, written, end, want in checks)


def variant_list_pairs(model, variant, split="heldout", num_pairs=None, seed=LIST_SEED):
    """The pairs of `list_pairs`, drawn and asked in the variant, with the same no-patch check. The
    checks are saved in dataset/, one file per variant. Held-out pairs get the extra source images."""
    first, last = SPLITS[split]
    if num_pairs is not None:
        last = min(last, first + num_pairs)
    cache_file = cache_path(model, f"list_{variant['name']}_seed{seed}.json")
    checked = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    pairs = []
    for pair in variant_list_candidates(variant, seed):
        if len(pairs) == last:
            break
        key = str(pair["candidate"])
        if key not in checked:
            checked[key] = _variant_list_passes(model, pair, variant)
            CACHE.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps(checked))
        if checked[key]:
            pair["index"] = len(pairs)
            pairs.append(pair)
    add_heldout_sources(pairs[SPLITS["heldout"][0]:])
    for pair in pairs:                                  # the extra sources share the target's canvas centers
        for name, side in pair.items():
            if isinstance(side, dict) and "shapes" in side and "canvas_centers" in pair["target"]:
                side["canvas_centers"] = pair["target"]["canvas_centers"]
    return pairs[first:last]


def list_pass_rate(variant, seed=LIST_SEED):
    """The share of list candidates that passed the no-patch check so far in this variant, in percent."""
    cache_file = cache_path(None, f"list_{variant['name']}_seed{seed}.json")
    checked = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    return 100 * sum(checked.values()) / max(1, len(checked))


def variant_progress_pairs(model, variant, sample, seed, num_pairs):
    """Pairs of one progress sampler (for example `sample_position_pair`) in the variant, drawn from
    one seed until `num_pairs` pass the no-patch check of pair["no_patch"]. Checks are saved in dataset/."""
    cache_file = cache_path(model, f"progress_{sample.__name__}_{variant['name']}_seed{seed}.json")
    checked = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    rng, pairs = random.Random(seed), []
    for draw in itertools.count():
        if len(pairs) == num_pairs:
            return pairs
        pair = sample(rng)
        if str(draw) not in checked:
            checked[str(draw)] = all(run_list_variant(model, pair[side], variant, record=False).answer.strip() == letter
                                     for side, letter in pair["no_patch"].items())
            CACHE.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps(checked))
        if checked[str(draw)]:
            pair["index"] = len(pairs)
            pairs.append(pair)


def list_layout_side(rng, ask, matching, letters, fillers=_FILLERS):
    """One side of a list image with six places, the shape we ask for at `matching` and a filler
    shape elsewhere (the shapes the model confuses with the asked one are left out)."""
    others = [s for s in fillers if s != ask and s not in CONFUSED[ask]]
    return {"shapes": [ask if p in matching else rng.choice(others) for p in range(6)], "letters": list(letters),
            "ask": ask, "matching": sorted(matching), "centers": None, "written": []}


# ------------------------------------------------------------------------------------ comic images
# The first version of the one-answer task (Appendix G.3.1). Each panel of the strip is a comic drawing
# under a white header of 160 pixels, and the header carries the shape with its letter. The comic
# drawings are not part of this repository. The folder must hold 500 four-panel comics as
#     <comics>/comic1/p1.png ... <comics>/comic1/p4.png,  ...,  <comics>/comic500/p4.png
# (any image size; each panel is resized to 384 x 384). The paper used 500 four-panel comic strips
# that we drew for it; any 500 four-panel comics in this layout run the same experiments.
COMIC_PROMPT = "Each panel has one shape marker containing one letter. What is the letter inside the {shape} shape?"
COMIC_SEED = 20260909
COMIC_HEADER = 160
COMIC_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
COMIC_SPLITS = {"search": (0, 100), "validation": (100, 150), "heldout": (150, 250)}


def comic_ids(seed=COMIC_SEED):
    """The comic numbers 1 to 500 in the paper's shuffled order. Pair i uses entries 2i and 2i + 1."""
    ids = list(range(1, 501))
    random.Random(seed).shuffle(ids)
    return ids


def comic_pair(index, seed=COMIC_SEED):
    """Comic pair number `index` (0 to 249). The shapes and questions are drawn as in
    `sample_one_answer_pair`, from all 26 letters, and the shapes sit in the header of each panel."""
    rng = random.Random(seed + 10_000 + index)
    shapes = rng.sample(SHAPES, 4)
    source_ask, target_ask = rng.sample(shapes, 2)
    source_order = rng.sample(shapes, 4)
    free = [i for i in range(4) if i not in (source_order.index(source_ask), source_order.index(target_ask))]
    rng.shuffle(free)
    target_order = [None] * 4
    target_order[free[0]], target_order[free[1]] = source_ask, target_ask
    others = [s for s in shapes if s not in (source_ask, target_ask)]
    rng.shuffle(others)
    for i, shape in zip([i for i, s in enumerate(target_order) if s is None], others):
        target_order[i] = shape
    letters = rng.sample(COMIC_LETTERS, 8)
    centers = [(rng.randint(72, 312), rng.randint(60, 100)) for _ in range(4)]
    ids = comic_ids(seed)
    return {"index": index,
            "source": {"shapes": source_order, "letters": letters[:4], "ask": source_ask, "centers": centers, "comic": ids[2 * index]},
            "target": {"shapes": target_order, "letters": letters[4:], "ask": target_ask, "centers": centers, "comic": ids[2 * index + 1]}}


def comic_pairs(split="heldout", num_pairs=None):
    """The 250 comic pairs split as in the paper: 100 search, 50 validation, 100 held-out pairs.
    Every pair is used; there is no no-patch filter in this version of the task."""
    first, last = COMIC_SPLITS[split]
    last = last if num_pairs is None else min(last, first + num_pairs)
    return [comic_pair(i) for i in range(first, last)]


def _comic_marker(draw, shape, letter, cx, cy, font):
    """The shape of the comic version: radius 53 for every shape, with a letter outlined in white."""
    r, style = 53, {"fill": "white", "outline": "black", "width": 6}
    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), **style)
    elif shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), **style)
    elif shape == "triangle":
        draw.polygon(_polygon(cx, cy + 4, r, 3), **style)
    elif shape == "diamond":
        draw.polygon(_polygon(cx, cy, r, 4), **style)
    elif shape == "star":
        draw.polygon(_star(cx, cy, r), **style)
    elif shape == "plus":
        draw.polygon(_plus(cx, cy, r, r * 0.36), **style)
    else:
        draw.polygon(_heart(cx, cy, r), **style)
    draw.text((cx, cy + (10 if shape == "triangle" else 1)), letter, font=font, fill="black", anchor="mm",
              stroke_width=1, stroke_fill="white")


def comic_strip(comics, comic):
    """A strip of the four panels of one comic under a white header, with panel borders. Returns
    the image; `image.regions` are the four full-height panels."""
    image = Image.new("RGB", (4 * PANEL, PANEL + COMIC_HEADER), "white")
    draw = ImageDraw.Draw(image)
    for p in range(4):
        with Image.open(Path(comics) / f"comic{comic}" / f"p{p + 1}.png") as panel:
            image.paste(panel.convert("RGB").resize((PANEL, PANEL), Image.Resampling.LANCZOS), (p * PANEL, COMIC_HEADER))
        draw.rectangle((p * PANEL, 0, (p + 1) * PANEL - 1, PANEL + COMIC_HEADER - 1), outline="#666666", width=3)
    image.regions = [[p * PANEL, 0, (p + 1) * PANEL, PANEL + COMIC_HEADER] for p in range(4)]
    return image


def comic_image(comics, side):
    """One side of a comic pair: the comic strip with the shapes and letters in the headers."""
    image = comic_strip(comics, side["comic"])
    draw, font = ImageDraw.Draw(image), ImageFont.truetype(str(FONT), 48)
    for p, (shape, letter, (cx, cy)) in enumerate(zip(side["shapes"], side["letters"], side["centers"])):
        _comic_marker(draw, shape, letter, p * PANEL + cx, cy, font)
    image.shapes, image.letters = list(side["shapes"]), list(side["letters"])
    return image


def run_comic(model, comics, side, record=True):
    from utils.patching import run_model
    run = run_model(model, comic_image(comics, side), COMIC_PROMPT.format(shape=side["ask"]), record=record)
    run.name_token("shape", run.find(" " + side["ask"]))
    run.name_token("colon", run.last)
    return run


def comic_both_directions(model, comics, pair):
    """`both_directions` on comic images."""
    a, b = run_comic(model, comics, pair["source"]), run_comic(model, comics, pair["target"])
    yield a, b, outcome_letters(pair["source"], pair["target"])
    yield b, a, outcome_letters(pair["target"], pair["source"])


# ------------------------------------------------------------------------------------ the colored-shapes task, other images
def _color_shape(draw, shape, color, cx, cy, r=72):
    """One filled shape of the colored-shapes task, as `draw_color_strip` draws it."""
    style = {"fill": {"cyan": CYAN, **OTHER_COLORS}.get(color, color), "outline": "black", "width": 1}
    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), **style)
    elif shape == "square":
        draw.polygon([(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)], **style)
    elif shape == "star":
        draw.polygon(_star(cx, cy, int(r * 1.3)), **style)
    else:
        draw.polygon(_heart(cx, cy - r // 8, int(r * 1.35)), **style)


def canvas_color_image(spec):
    """A colored-shapes image (see `sample_color_image`) with the four shapes at random places on one
    768 x 768 white canvas and no panels. The centers depend on the image index only."""
    centers = sample_centers(random.Random(70_000 + spec["index"]), 4, 768)
    image = Image.new("RGB", (768, 768), "white")
    draw = ImageDraw.Draw(image)
    for shape, color, (cx, cy) in zip(spec["shapes"], spec["colors"], centers):
        _color_shape(draw, shape, color, cx, cy)
    image.regions, image.shapes = [place_box(c) for c in centers], list(spec["shapes"])
    return image


def comic_color_image(comics, spec):
    """A colored-shapes image drawn over a comic strip: the four colored shapes sit at the top of the
    four panels, over the header and the comic. Image i uses the comic in place i of `comic_ids`."""
    image = comic_strip(comics, comic_ids()[spec["index"] % 500])
    draw = ImageDraw.Draw(image)
    for p, (shape, color) in enumerate(zip(spec["shapes"], spec["colors"])):
        _color_shape(draw, shape, color, p * PANEL + PANEL // 2, 128)
    image.shapes = list(spec["shapes"])
    return image


def color_cases_drawn(model, spec, draw, target_languages=("de", "fr", "zh")):
    """`color_cases` with another drawing of the images: `draw(spec)` returns the image."""
    from utils.patching import run_model
    ask = lambda s, language, record=True: run_model(model, draw(s), COLOR_QUESTIONS[language], prefill="", record=record)
    targets = {language: ask(spec, language, record=False) for language in target_languages}
    for source_spec in spec["sources"]:
        sources = {language: ask(source_spec, language) for language in LANGUAGES}
        for target_language in target_languages:
            for source_language in LANGUAGES:
                if source_language != target_language:
                    yield sources[source_language], targets[target_language], {
                        "source_image": source_spec["index"], "source_language": source_language,
                        "target_language": target_language, "source_shape": source_spec["shape"], "target_shape": spec["shape"]}


# ------------------------------------------------------------------------------------ shapes of different sizes on a canvas
SIZE_SEED = 20261112
SIZE_RADIUS = {"circle": 72, "square": 68, "triangle": 96, "diamond": 88, "star": 100, "plus": 88, "heart": 92}
SIZE_SCALES = {4: [0.6, 0.85, 1.15, 1.5], 5: [0.6, 0.76, 0.95, 1.2, 1.5]}   # smallest to largest
SIZE_GLYPHS = {4: [24, 33, 44, 58], 5: [24, 30, 38, 48, 60]}                # letter sizes by size rank
SIZE_VERSIONS = ("shapes_and_letters", "shapes_only", "letters_only", "one_size")


def size_layouts(num_layouts=100, seed=SIZE_SEED):
    """Layouts of six shapes on a 960 x 960 canvas with the asked shape at four (even index) or five
    (odd index) places. The asked shapes get clearly different sizes, each at least 1.5 times the area
    of the next smaller one, in a random order that does not depend on the place."""
    rng, layouts = random.Random(seed), []
    for index in range(num_layouts):
        n, ask = (4 if index % 2 == 0 else 5), LIST_ASK[index % 3]
        others = [s for s in SHAPES if s != ask and s not in CONFUSED[ask]]
        places = sorted(rng.sample(range(6), n))
        shapes = [ask if p in places else rng.choice(others) for p in range(6)]
        letters = rng.sample(LETTERS, 6)
        ranks = rng.sample(range(n), n)                                   # size rank of each asked place (0 = smallest)
        rank = [None] * 6
        scale = [0.0] * 6
        for k, p in enumerate(places):
            rank[p], scale[p] = ranks[k], SIZE_SCALES[n][ranks[k]]
        for p in range(6):
            if p not in places:
                scale[p] = rng.choice(SIZE_SCALES[n])
        radii = [SIZE_RADIUS[s] * max(scale[p], 1.0) for p, s in enumerate(shapes)]
        layouts.append({"index": index, "ask": ask, "places": places, "shapes": shapes, "letters": letters,
                        "size_rank": rank, "scale": scale, "centers": _sized_centers(rng, radii)})
    return layouts


def _sized_centers(rng, radii, side=960, gap=36, x_gap=64):
    """Centers for shapes of these radii: shapes keep `gap` pixels apart and differ by `x_gap` in x."""
    for _ in range(20_000):
        points = {}
        for i in rng.sample(range(len(radii)), len(radii)):
            r = radii[i]
            for _ in range(500):
                c = (rng.randint(int(r + PLACE_MARGIN), int(side - r - PLACE_MARGIN)),
                     rng.randint(int(r + PLACE_MARGIN), int(side - r - PLACE_MARGIN)))
                if all(math.dist(c, b) >= r + radii[j] + gap and abs(c[0] - b[0]) >= x_gap for j, b in points.items()):
                    points[i] = c
                    break
            else:
                break
        if len(points) == len(radii):
            return [list(points[i]) for i in range(len(radii))]
    raise RuntimeError("could not place the shapes")


def size_image(layout, version):
    """One layout in one of SIZE_VERSIONS: shapes and letters of different sizes, only the shapes,
    only the letters, or one size for all (shape scale 1, letters 30 pixels)."""
    image = Image.new("RGB", (960, 960), "white")
    draw = ImageDraw.Draw(image)
    for p, (shape, letter, (cx, cy)) in enumerate(zip(layout["shapes"], layout["letters"], layout["centers"])):
        s, rank = layout["scale"][p], layout["size_rank"][p]
        if version == "shapes_and_letters":
            scale, glyph = s, round(40 * s)
        elif version == "shapes_only":
            scale, glyph = s, 30
        elif version == "letters_only":
            scale, glyph = 1.0, SIZE_GLYPHS[len(layout["places"])][rank] if rank is not None else 30
        else:
            scale, glyph = 1.0, 30
        r = SIZE_RADIUS[shape] * scale
        style = {"fill": "white", "outline": "black", "width": 6}
        if shape == "circle":
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), **style)
        elif shape == "square":
            draw.rectangle((cx - r, cy - r, cx + r, cy + r), **style)
        elif shape == "triangle":
            draw.polygon(_polygon(cx, cy + 10 * scale, r, 3), **style)
        elif shape == "diamond":
            draw.polygon(_polygon(cx, cy, r, 4), **style)
        elif shape == "star":
            draw.polygon(_star(cx, cy + 4 * scale, r), **style)
        elif shape == "plus":
            draw.polygon(_plus(cx, cy, r, r * 0.42), **style)
        else:
            draw.polygon(_heart(cx, cy, r), **style)
        dy = {"triangle": 22, "star": 8, "heart": 16}.get(shape, 1) * scale
        draw.text((cx, cy + dy), letter, font=ImageFont.truetype(str(FONT), glyph), fill="black", anchor="mm")
    return image


# ------------------------------------------------------------------------------------ order questions on photographs
COCO_ORDER_SENTENCES = {"neutral": " List them in any order.", "most_salient": " List them starting with the most salient one.",
                        "most_prominent": " Start with the most prominent one.", "left_to_right": " List them from left to right."}


def coco_written_list(text, categories):
    """The categories a written list names, in order, each once. `categories` are the annotated ones."""
    named = []
    for item in text.split(","):
        category = COCO_CATEGORY_OF_WORD.get(item.strip().lower().rstrip("."))
        if category in categories and category not in named:
            named.append(category)
    return named


def coco_category_features(entry):
    """Per annotated category of a COCO entry: its center x and y (area-weighted over its boxes,
    as a fraction of the image), its area (share of the image covered by the union of its boxes, on a
    128 x 128 grid), and its distance from the image center (0 at the center, 1 at a corner)."""
    import numpy as np
    width, height, features = entry["width"], entry["height"], {}
    for category, boxes in entry["boxes"].items():
        mask = np.zeros((128, 128), dtype=bool)
        cx = cy = total = 0.0
        for x, y, w, h in boxes:
            cx, cy, total = cx + (x + w / 2) / width * w * h, cy + (y + h / 2) / height * w * h, total + w * h
            mask[int(y / height * 128):max(math.ceil((y + h) / height * 128), int(y / height * 128) + 1),
                 int(x / width * 128):max(math.ceil((x + w) / width * 128), int(x / width * 128) + 1)] = True
        cx, cy = cx / max(total, 1e-9), cy / max(total, 1e-9)
        features[category] = {"x": cx, "y": cy, "area": float(mask.mean()),
                              "distance": math.hypot(cx - 0.5, cy - 0.5) / math.hypot(0.5, 0.5)}
    return features


def coco_rules(first, second):
    """Which of two named categories each rule puts first: +1 if `first`, -1 if `second`, 0 if the rule
    cannot order them (sizes within a factor of 1.25, or positions within 0.05 of the image). The
    features come from `coco_category_features`."""
    def close(a, b, tolerance):
        return 0 if abs(a - b) < tolerance else (1 if a < b else -1)
    def bigger(a, b):
        ratio = a / max(b, 1e-9)
        return 0 if 1 / 1.25 < ratio < 1.25 else (1 if ratio > 1 else -1)
    salience = lambda f: f["area"] * (1 - f["distance"])          # size times closeness to the center
    return {"larger": bigger(first["area"], second["area"]), "size_times_centrality": bigger(salience(first), salience(second)),
            "more_central": close(first["distance"], second["distance"], 0.05),
            "left": close(first["x"], second["x"], 0.05), "upper": close(first["y"], second["y"], 0.05)}


# ------------------------------------------------------------------------------------ the right-to-left instruction
RIGHT_TO_LEFT = WORDINGS["right_to_left"][1][len(LIST_PROMPT):]        # " List them from right to left."


def instruction_positions(run):
    """The positions of the tokens of " List them from right to left." in a run that has it."""
    tokens = run.tokenizer.encode(RIGHT_TO_LEFT, add_special_tokens=False)
    starts = [i for i in range(len(run.ids)) if run.ids[i:i + len(tokens)] == tokens]
    assert len(starts) == 1, "the instruction must appear once"
    return list(range(starts[0], starts[0] + len(tokens)))


def sample_row_list_image(rng):
    """A row of six panels with the asked shape in four of them, filler shapes elsewhere."""
    side = list_layout_side(rng, rng.choice(LIST_ASK), sorted(rng.sample(range(6), 4)), [""] * 6)
    side["letters"] = rng.sample(LETTERS, 6)
    return side


def right_to_left_items(model, seed, count):
    """Yields (R, D) for `count` row images from one seed. R is the run with the right-to-left
    instruction and the letter of r2 written, where r1 to r4 are the matching panels from right to
    left, so R answers r3. D is the run with the paper's prompt, the same image and the same written
    letter. An image is used when R answers r3 and D answers something else."""
    right_to_left, default = make_variant("row", "right_to_left"), make_variant("row", "paper")
    rng, found = random.Random(seed), 0
    while found < count:
        side = sample_row_list_image(rng)
        r = [side["letters"][p] for p in side["matching"][::-1]]              # the letters at r1 .. r4
        R = run_list_variant(model, side, right_to_left, [r[1]], record=False)
        D = run_list_variant(model, side, default, [r[1]], record=False)
        if R.answer == " " + r[2] and D.answer != R.answer:
            found += 1
            yield R, D
