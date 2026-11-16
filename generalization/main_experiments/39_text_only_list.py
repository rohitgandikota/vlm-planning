"""A list with no image: the panels described in words (Appendix G.3.7).

Qwen3-VL-8B gets no image. The six panels are described in text ("Six panels from left to right.
Panel 1: a star with the letter G. Panel 2: a heart with the letter D. ..."), followed by the paper's
list question (utils.dataset.panel_descriptions, utils.patching.run_model_text). Three tests:
    1. Lists. The asked shape is in 2 to 4 of the six panels. Does the model list the right letters?
    2. The comma token cannot attend to the answer token (progress/main_experiments/06, on the list
       pairs described in words): the model writes the same letter again. Control: the colon token.
    3. Progress at the comma token (progress/main_experiments/02, on its pairs described in words):
       the source's last comma residual replaces the target's colon residual at one layer, and the
       target gives its second answer. Control: a source with nothing written gives its colon residual.

    python generalization/main_experiments/39_text_only_list.py

--num_pairs is the number of items of each test (the paper used 200 lists and 100 items for 2 and
3). Pairs are kept when their no-patch runs, described in words, are right.

Paper: the model lists the letters correctly in 89% of items. With the comma blocked from the
answer token it repeats the written letter in 72% of cases, against 99 to 100% correct with the
colon blocked. The progress test gives 86% at layer 21 (84% at 20), against 0% for the source with
nothing written.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import (LETTERS, LIST_ASK, SHAPES, list_layout_side, list_question, make_variant, matching_letters,
                           name_answer, panel_descriptions, run_list_variant, sample_progress_pair, variant_list_pairs,
                           variant_progress_pairs)
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import Patch, make_patch, run_model_text, run_patches, write_tokens


def main():
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    text, every_layer, layers = make_variant("text"), range(n_layers(model)), range(n_layers(model) + 1)
    results = Results("text_only_list", args)
    rng = random.Random(1700)
    for index in range(2 * args.num_pairs):                                      # 1. lists
        side = list_layout_side(rng, rng.choice(LIST_ASK), sorted(rng.sample(range(6), rng.randint(2, 4))), [""] * 6, fillers=SHAPES)
        side["letters"] = rng.sample(LETTERS, 6)
        run = run_model_text(model, panel_descriptions(side), list_question(side["ask"]), record=False)
        written = [item.strip() for item in write_tokens(model, run, max_tokens=16).split(",") if item.strip()]
        results.add(f"list{index}", list_right=written == matching_letters(side))
    for pair in variant_list_pairs(model, text, args.split, args.num_pairs):        # 2. comma blocked
        target = run_list_variant(model, pair["target"], text, [pair["written"]])
        comma = [target.token("last_comma")]
        blocked, control = run_patches(model, target, [Patch().block(None, comma, [target.token("last_answer")], layers=every_layer),
                                                       Patch().block(None, comma, [target.token("colon")], layers=every_layer)])["answers"]
        results.add(pair["index"], same_letter_again=blocked == " " + pair["written"],
                    colon_blocked_unchanged=control == " " + matching_letters(pair["target"])[1])
    for pair in variant_progress_pairs(model, text, sample_progress_pair, seed=510, num_pairs=args.num_pairs):   # 3. progress
        source, target = run_list_variant(model, pair["source"], text), run_list_variant(model, pair["target"], text)
        nothing_written = run_list_variant(model, pair["source"], text, written=[])
        patches = [make_patch(source, target, "residual", "colon", source_token="last_comma", layer=layer) for layer in layers]
        patches += [make_patch(nothing_written, target, "residual", "colon", layer=layer) for layer in layers]
        outcomes = [name_answer(answer, pair["letters"]) for answer in run_patches(model, target, patches)["answers"]]
        results.add(f"progress{pair['index']}", **{f"second_layer_{layer}": outcomes[layer] == "second" for layer in layers},
                    **{f"nothing_written_second_layer_{layer}": outcomes[len(layers) + layer] == "second" for layer in layers})
    results.report("list_right", paper=89.0, label="lists right")
    results.report("same_letter_again", paper=72.0, label="comma blocked from the answer token: same letter again")
    results.report("colon_blocked_unchanged", paper="99 to 100", label="colon token blocked: answer unchanged")
    results.report("second_layer_20", paper=84.0, label="progress at layer 20: second answer")
    results.report("second_layer_21", paper=86.0, label="progress at layer 21: second answer")
    results.report("nothing_written_second_layer_21", paper=0.0, label="nothing written, layer 21: second answer")
    results.save()


if __name__ == "__main__":
    main()
