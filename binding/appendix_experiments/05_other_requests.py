"""Binding heads bind any request word (Appendix B.5, Table "other-bindings").

Every test of Section 4 names a shape. Do the binding heads bind shape words, or any request? We
change what the question names and keep the paper's head sets unchanged: the same binding, query,
key, relay and gaze heads, with no new search. Four variants of the four-panel task (see
`OTHER_REQUESTS` in utils/dataset.py):
    color      "the letter inside the red circle"         the request word is the color word
    category   "the letter next to the cat"               a drawn object with a letter beside it
    two        "the letter inside the red star"           two request words; the patches at the
                                                          request tokens act at the color word, at the
                                                          shape word, and at both
    words      the paper's shapes with three-letter words; under the paper's question the model
               writes the word's initial, so we score the initial
    wordq      the same images, with a question that asks for the word; we score the word
On each variant we repeat the patches of Sections 4 and 6. Each patch has random heads from the
same layers as its control.

Each variant draws its own 500 pairs from its own seed (150), not the standard held-out pairs. A
pair is kept if the model answers both questions correctly with no patch. The category variant
needs a color emoji font (Noto Color Emoji; set EMOJI_FONT to its path).

    python binding/appendix_experiments/05_other_requests.py --num_pairs 500                  # all five
    python binding/appendix_experiments/05_other_requests.py --num_pairs 500 --variant color

Paper: the patches work in 75 to 100% of cases, random heads in at most 4.0% (values in PAPER).
With two words, the queries at the color word alone make the target name the item with the
source's color and its own shape in 78.6%, the queries at the shape word alone the item with its
own color and the source's shape in 85.9%; the outputs move the pointer in 72.4% at the shape word,
23.8% at the color word, and 93.8% at both. With word labels and the word question, the gaze heads'
image values give the source's word in 1.4% and its initial in 15.8% more; when only the shape
token and the gaze heads see the image, the model writes its own word in 16.1% and its initial in 79.9%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import OTHER_REQUESTS, is_label, other_request_directions, other_request_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import all_heads, load_model, n_layers
from utils.patching import Patch, make_patch, record_patch, run_patches

# The table of the paper, per variant: color, category, two, words, wordq.
PAPER = {"queries": (98.9, 82.1, 84.4, 85.0, 86.2), "keys": (99.8, 97.5, 81.5, 80.9, 86.1),
         "queries_and_keys": (100.0, 99.9, 93.9, 87.8, 97.0), "outputs": (96.9, 96.9, 93.8, 75.4, 85.8),
         "relay": (97.9, 95.3, 94.3, 79.9, 86.3), "restore": (100.0, 100.0, 100.0, 99.6, 100.0),
         "gaze_queries": (92.1, 92.0, 88.5, 59.7, 74.8), "gaze_values": (99.6, 99.8, 99.1, 87.9, 1.4),
         "only_gaze_see": (100.0, 99.9, 99.8, 99.6, 16.1), "colon_layer24": (95.3, 91.6, 93.4, 46.1, 74.2),
         "native": (100.0, 100.0, 100.0, 95.4, None)}
EXTRA = {"two": {"queries_color_word_color_swap": 78.6, "queries_shape_word_shape_swap": 85.9,
                 "outputs_shape_word": 72.4, "outputs_color_word": 23.8},
         "wordq": {"gaze_values_initial": 15.8, "only_gaze_see_initial": 79.9}}


def patch_at(run, target, components, heads, tokens):
    """One patch that copies `components` of `heads` from `run` into the target at every token in `tokens`."""
    patch = Patch()
    for component in components:
        for token in tokens:
            patch.add(make_patch(run, target, component, token, heads=heads))
    return patch


def case_patches(model, source, target, seed):
    """Every patch of one case, as {name: (patch, the outcome letter it should give)}."""
    words, colon = target.token("request"), target.token("colon")
    sites = {"": words} if len(words) == 1 else {"_color_word": words[:1], "_shape_word": words[1:], "": words}
    patches = {}
    for tag, control in (("", False), ("_random", True)):
        pick = (lambda name: random_like(HEADS[name], seed=seed)) if control else (lambda name: HEADS[name])
        for site, tokens in sites.items():
            for name, components, heads, letter in (("queries", ["query"], "query", "P"), ("keys", ["image_keys"], "key", "R"),
                                                    ("queries_and_keys", ["query", "image_keys"], "query_and_key", "K"),
                                                    ("outputs", ["head_output"], "binding", "K")):
                patches[name + site + tag] = (patch_at(source, target, components, pick(heads), tokens), letter)
        gaze = pick("gaze")
        patches["gaze_queries" + tag] = (make_patch(source, target, "query", "colon", heads=gaze), "K")
        patches["gaze_values" + tag] = (make_patch(source, target, "image_values", "colon", heads=gaze), "C")
        only = Patch().block(None, range(words[-1] + 1, colon), target.image_positions, layers=range(n_layers(model)))
        patches["only_gaze_see" + tag] = (only.block([h for h in all_heads(model) if h not in gaze], [colon],
                                                     target.image_positions), "T")
    # the relay heads read the values of a run with the binding-head patch at every request word
    binding = lambda: patch_at(source, target, ["head_output"], HEADS["binding"], words)
    induced = record_patch(model, target, binding())
    read_from = range(words[0], colon)
    for tag, relay in (("", HEADS["relay"]), ("_random", random_like(HEADS["relay"], seed=seed))):
        patches["relay" + tag] = (make_patch(induced, target, "values", "colon", heads=relay, positions=read_from), "K")
    patches["restore"] = (binding().add(make_patch(target, target, "values", "colon", heads=HEADS["relay"],
                                                   positions=read_from)), "T")
    for layer in range(n_layers(model) + 1):
        patches[f"colon_layer{layer}"] = (make_patch(source, target, "residual", "colon", layer=layer), "K")
    return patches


def named_item(answer, source, target):
    """Variant two: which target item an answer letter names, by color and shape."""
    item = dict(zip(target["letters"], target["shapes"])).get(answer.strip())
    if item is None or item in (target["ask"], source["ask"]):
        return item
    return "color_swap" if item.split()[0] == source["ask"].split()[0] else "shape_swap"


def run_variant(model, args, variant):
    column = OTHER_REQUESTS.index(variant)
    results = Results(f"other_requests_{variant}", args)
    pairs, tried = other_request_pairs(model, variant, args.start + args.num_pairs)
    for draw, sides_right in tried.items():
        for right in sides_right:
            results.add(draw, native=right)
    for pair in pairs[args.start:]:
        sides = [(pair["source"], pair["target"]), (pair["target"], pair["source"])]      # the same two sides, as dicts
        runs = other_request_directions(model, variant, pair)
        for (source, target, letters), (source_side, target_side) in zip(runs, sides):
            patches = case_patches(model, source, target, seed=1000 * pair["index"] + 7)
            answers = dict(zip(patches, run_patches(model, target, [p for p, _ in patches.values()])["answers"]))
            row = {name: is_label(answers[name], letters[letter], variant) for name, (_, letter) in patches.items()}
            if variant == "two":
                for word, swap in (("color_word", "color_swap"), ("shape_word", "shape_swap")):
                    row[f"queries_{word}_{swap}"] = named_item(answers[f"queries_{word}"], source_side, target_side) == swap
            if variant == "wordq":         # a single letter that is the initial of the expected word
                row["gaze_values_initial"] = answers["gaze_values"].strip() == letters["C"][0]
                row["only_gaze_see_initial"] = answers["only_gaze_see"].strip() == letters["T"][0]
            results.add(pair["index"], **row)
    print(f"\n== {variant}")
    for name, values in PAPER.items():
        results.report(name, paper=values[column])
        if any(name + "_random" in row for row in results.rows):
            results.report(name + "_random", paper=None if name == "only_gaze_see" else "at most 4.0", label="  random heads")
    for name, paper in EXTRA.get(variant, {}).items():
        results.report(name, paper=paper)
    results.save()


def main():
    variant = "all"
    if "--variant" in sys.argv:
        at = sys.argv.index("--variant")
        variant = sys.argv.pop(at + 1)
        sys.argv.pop(at)
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    for name in OTHER_REQUESTS if variant == "all" else [variant]:
        run_variant(model, args, name)


if __name__ == "__main__":
    main()
