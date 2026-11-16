"""The heads of the paper on other images and other prompts: the one-answer task (Appendix G.3.2, G.3.4).

We change the layout of the image or the wording of the prompt, one at a time, and run the
one-answer patches of Sections 4 and 6 with the heads found on the paper's row of panels. No head
is searched again. Each patch is scored as in its main experiment:
    binding queries at the shape token            P   (the source's request)
    binding heads read the source's image keys    R   (the source's layout)
    queries and keys together, binding outputs    K   (the source's panel)
    relay values from the run with the binding-head patch, at the colon token     K
    the same with the target's own relay values restored                           T
    gaze-head queries at the colon token          K;   pointer moved with gaze queries held    T
    gaze heads read the source's image values     C;   only the gaze heads see the image       T
    blocks: relay heads from the text at the colon token, later tokens from the shape token,
    binding heads from the image at the shape token, gaze heads from the image at the colon token
                                                  T   (the target keeps its own answer)
    the residual of the colon token, at each layer                                 K   (best layer)

    python generalization/main_experiments/26_one_answer_on_other_images.py --layout canvas
    python generalization/main_experiments/26_one_answer_on_other_images.py --layout grid
    python generalization/main_experiments/26_one_answer_on_other_images.py --layout column
    python generalization/main_experiments/26_one_answer_on_other_images.py --layout photo --coco /path/to/coco
    python generalization/main_experiments/26_one_answer_on_other_images.py --wording one_sentence_after
    python generalization/main_experiments/26_one_answer_on_other_images.py --wording two_sentences_after

Layouts: shapes on a blank canvas with no panels, a 2 x 2 grid, a column, or shapes drawn over a
COCO photograph (the first 250 images of val2014; a place is the 248-pixel box around a shape).
Pairs: the paper's pair generator, drawn in the variant, kept if both questions are answered right
in the variant. The paper's numbers (below) come from an earlier, smaller run: 100 held-out pairs on
the canvas, 30 to 60 on the other variants, with that run's own pairs and head sets, so expect values
near them, not the same values. The default here is 100 pairs.

Paper (tab:variants-one, the top of tab:variants-gaze, and fig:appendix_no_panels_tests a): the
value of every patch for every variant is in PAPER below and is printed next to each result. For
example, the binding outputs move the pointer in 97.5% of cases on the canvas, 93.8% on the grid,
96.7% on photographs, and 53.3% with two sentences after the shape token. Random heads: at most 1.2%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import variant_argument, variant_both_directions, variant_one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import all_heads, load_model, n_layers
from utils.patching import Patch, make_patch, record_patch, run_patches

# The paper's value for each patch, by variant (None: not reported for that variant).
PAPER = {
    "canvas_paper": {"binding_outputs": 97.5, "relay_values": 95.5, "gaze_queries": 90.0, "gaze_queries_held": 99.0,
                     "only_gaze_heads_see": 98.0, "colon_residual_best_layer": 93.5},
    "row_one_sentence_after": {"binding_queries": 73.3, "binding_keys": 83.3, "queries_and_keys": 90.0, "binding_outputs": 80.0,
                               "relay_values": 79.2, "relay_values_restored": 99.2, "no_patch": 99.2, "relay_heads_blocked": 9.2,
                               "later_tokens_blocked": 30.8, "binding_heads_blind": 89.2, "gaze_heads_blind": 97.5,
                               "gaze_queries": 83.3, "gaze_queries_held": 98.3, "gaze_image_values": 94.2, "only_gaze_heads_see": 90.0},
    "row_two_sentences_after": {"binding_queries": 48.3, "binding_keys": 61.7, "queries_and_keys": 56.7, "binding_outputs": 53.3,
                                "relay_values": 58.3, "relay_values_restored": 93.3, "no_patch": 86.7, "relay_heads_blocked": 10.0,
                                "later_tokens_blocked": 25.0, "binding_heads_blind": 93.3, "gaze_heads_blind": 91.7,
                                "gaze_queries": 76.7, "gaze_queries_held": 93.3, "gaze_image_values": 95.0, "only_gaze_heads_see": 91.7},
    "grid_paper": {"binding_queries": 67.5, "binding_keys": 75.0, "queries_and_keys": 93.8, "binding_outputs": 93.8,
                   "relay_values": 93.8, "relay_values_restored": 98.8, "no_patch": 97.5, "relay_heads_blocked": 15.0,
                   "later_tokens_blocked": 25.0, "binding_heads_blind": 77.5, "gaze_heads_blind": 96.2,
                   "gaze_queries": 82.5, "gaze_queries_held": 97.5, "gaze_image_values": 90.0, "only_gaze_heads_see": 96.2},
    "column_paper": {"binding_queries": 72.5, "binding_keys": 75.0, "queries_and_keys": 92.5, "binding_outputs": 85.0,
                     "relay_values": 86.2, "relay_values_restored": 96.2, "no_patch": 97.5, "relay_heads_blocked": 21.2,
                     "later_tokens_blocked": 23.8, "binding_heads_blind": 82.5, "gaze_heads_blind": 95.0,
                     "gaze_queries": 73.8, "gaze_queries_held": 96.2, "gaze_image_values": 90.0, "only_gaze_heads_see": 96.2},
    "photo_paper": {"binding_queries": 90.8, "binding_keys": 98.3, "queries_and_keys": 96.7, "binding_outputs": 96.7,
                    "relay_values": 98.3, "relay_values_restored": 100.0, "no_patch": 100.0, "relay_heads_blocked": 20.0,
                    "later_tokens_blocked": 18.3, "binding_heads_blind": 78.3, "gaze_heads_blind": 100.0},
}


def binding_patches(source, target, heads):
    """The four binding patches at the shape token, for one set of (binding, query, key, both) heads."""
    query_and_keys = make_patch(source, target, "query", "shape", heads=heads["query_and_key"])
    return [make_patch(source, target, "query", "shape", heads=heads["query"]),
            make_patch(source, target, "image_keys", "shape", heads=heads["key"]),
            query_and_keys.add(make_patch(source, target, "image_keys", "shape", heads=heads["query_and_key"])),
            make_patch(source, target, "head_output", "shape", heads=heads["binding"])]


def relay_values(run, target, heads):
    """At the colon token, `heads` read the values of `run` from the shape token through the token before the colon token."""
    return make_patch(run, target, "values", "colon", heads=heads, positions=range(target.token("shape"), target.token("colon")))


def only_these_see_the_image(model, target, heads):
    """Every token after the shape token is blocked from the image, except `heads` at the colon token."""
    shape, colon = target.token("shape"), target.token("colon")
    others = [head for head in all_heads(model) if head not in heads]
    patch = Patch().block(None, range(shape + 1, colon), target.image_positions, layers=range(n_layers(model)))
    return patch.block(others, [colon], target.image_positions)


def main():
    variant = variant_argument()
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    layers = range(n_layers(model) + 1)
    results = Results(f"one_answer_on_other_images_{variant['name']}", args)
    paper_heads = {name: HEADS[name] for name in ("query", "key", "query_and_key", "binding")}
    for pair in variant_one_answer_pairs(model, variant, args.split, args.start + args.num_pairs)[args.start:]:
        seed = pair["index"]
        random_heads = {name: random_like(heads, seed=seed) for name, heads in paper_heads.items()}
        random_relay, random_gaze = random_like(HEADS["relay"], seed=seed), random_like(HEADS["gaze"], seed=seed)
        for source, target, letters in variant_both_directions(model, pair, variant):
            shape, colon, image, every_layer = target.token("shape"), target.token("colon"), target.image_positions, range(n_layers(model))
            binding = make_patch(source, target, "head_output", "shape", heads=HEADS["binding"])
            patched = record_patch(model, target, binding)                    # the run with the binding-head patch
            patches = binding_patches(source, target, paper_heads) + [
                relay_values(patched, target, HEADS["relay"]),
                make_patch(source, target, "head_output", "shape", heads=HEADS["binding"]).add(relay_values(target, target, HEADS["relay"])),
                Patch(),
                Patch().block(HEADS["relay"], [colon], range(shape, colon)),
                Patch().block(None, range(shape + 1, colon + 1), [shape], layers=every_layer),
                Patch().block(HEADS["binding"], [shape], image),
                Patch().block(HEADS["gaze"], [colon], image),
                make_patch(source, target, "query", "colon", heads=HEADS["gaze"]),
                make_patch(source, target, "head_output", "shape", heads=HEADS["binding"]).add(
                    make_patch(target, target, "query", "colon", heads=HEADS["gaze"])),
                make_patch(source, target, "image_values", "colon", heads=HEADS["gaze"]),
                only_these_see_the_image(model, target, HEADS["gaze"])]
            controls = binding_patches(source, target, random_heads) + [
                relay_values(patched, target, random_relay), make_patch(source, target, "query", "colon", heads=random_gaze)]
            sweep = [make_patch(source, target, "residual", "colon", layer=layer) for layer in layers]
            answers = run_patches(model, target, patches + controls + sweep + [only_these_see_the_image(model, target, random_gaze)])["answers"]
            a = dict(zip(["binding_queries", "binding_keys", "queries_and_keys", "binding_outputs", "relay_values",
                          "relay_values_restored", "no_patch", "relay_heads_blocked", "later_tokens_blocked", "binding_heads_blind",
                          "gaze_heads_blind", "gaze_queries", "gaze_queries_held", "gaze_image_values", "only_gaze_heads_see"], answers))
            is_ = lambda answer, outcome: answer == " " + letters[outcome]
            control_answers = answers[len(patches):len(patches) + len(controls)]
            residual = answers[len(patches) + len(controls):-1]
            results.add(pair["index"], binding_queries=is_(a["binding_queries"], "P"), binding_keys=is_(a["binding_keys"], "R"),
                        queries_and_keys=is_(a["queries_and_keys"], "K"), binding_outputs=is_(a["binding_outputs"], "K"),
                        relay_values=is_(a["relay_values"], "K"), relay_values_restored=is_(a["relay_values_restored"], "T"),
                        no_patch=is_(a["no_patch"], "T"), relay_heads_blocked=is_(a["relay_heads_blocked"], "T"),
                        later_tokens_blocked=is_(a["later_tokens_blocked"], "T"), binding_heads_blind=is_(a["binding_heads_blind"], "T"),
                        gaze_heads_blind=is_(a["gaze_heads_blind"], "T"), gaze_queries=is_(a["gaze_queries"], "K"),
                        gaze_queries_held=is_(a["gaze_queries_held"], "T"), gaze_image_values=is_(a["gaze_image_values"], "C"),
                        only_gaze_heads_see=is_(a["only_gaze_heads_see"], "T"),
                        random_heads_any_patch=any(is_(x, o) for x, o in zip(control_answers, "PRKKKK")),
                        random_only_see=is_(answers[-1], "T"),
                        **{f"colon_residual_layer_{layer}": is_(x, "K") for layer, x in zip(layers, residual)})
    paper = PAPER.get(variant["name"], {})
    for condition in ("binding_queries", "binding_keys", "queries_and_keys", "binding_outputs", "relay_values",
                      "relay_values_restored", "no_patch", "relay_heads_blocked", "later_tokens_blocked", "binding_heads_blind",
                      "gaze_heads_blind", "gaze_queries", "gaze_queries_held", "gaze_image_values", "only_gaze_heads_see"):
        results.report(condition, paper=paper.get(condition))
    results.report("random_heads_any_patch", paper="at most 1.2", label="random heads, any patch")
    results.report("random_only_see", paper="at most 3.8", label="only random heads see the image")
    best = max(layers, key=lambda layer: results.rate(f"colon_residual_layer_{layer}")["percent"] or 0)
    results.report(f"colon_residual_layer_{best}", paper=paper.get("colon_residual_best_layer"),
                   label=f"colon residual, best layer ({best}): pointer moved (K)")
    results.save(variant=variant["name"])


if __name__ == "__main__":
    main()
