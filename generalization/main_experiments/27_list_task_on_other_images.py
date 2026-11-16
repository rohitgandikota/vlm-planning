"""The heads of the paper on other images and other prompts: the list task (Appendix G.3.2, G.3.4).

The list pairs of the paper (utils.dataset.list_pairs), drawn and asked in a variant, with the heads
found on the paper's row of panels. r1 to r4 are the target's matching panels and W its letter at r1.
    First answer (nothing written; the source has its first matching panel at r2):
        binding outputs at the shape token, relay values at the colon token, gaze queries at the
        colon token                                       -> the target's letter at r2
    Second answer ("Answer: W," written; the source's next matching panel is r4):
        gaze queries at the last comma token              -> the target's letter at r4
        the same from a source whose next panel is a written panel, or a panel that holds
        another shape in the target (held-out pairs only)  -> the target's letter there
        the comma token cannot attend to the answer token (every head and layer)  -> W again
        control: the comma token cannot attend to the colon token                  -> W again
        the relay heads cannot attend to the text at the comma token               -> not its own answer
        the answer token and the comma token blocked from the image                -> not its own answer
    Stopping ("Answer: W" on the image whose only matching panel is r1, which ends the list):
        the shape token blocked from the image in every layer                      -> the list goes on

    python generalization/main_experiments/27_list_task_on_other_images.py --layout canvas
    python generalization/main_experiments/27_list_task_on_other_images.py --layout grid
    python generalization/main_experiments/27_list_task_on_other_images.py --layout column
    python generalization/main_experiments/27_list_task_on_other_images.py --wording one_sentence_after
    python generalization/main_experiments/27_list_task_on_other_images.py --wording two_sentences_after

On a canvas the places are numbered in the model's reading order, and only layouts with a clear
order are used (utils.dataset.sample_ordered_centers). Pairs are kept if every no-patch run is right
in the variant. The paper's numbers come from an earlier, smaller run (40 to 60 pairs per variant,
200 on the canvas), so expect values near them, not the same values. The default here is 100 pairs.

Paper (bottom of tab:variants-gaze, fig:appendix_no_panels_tests b), for example: relay values move
the first answer in 83.2% on the canvas, 93.1% with one sentence after the shape token, 97.4% on the
grid; gaze queries at the comma move the second answer in 86.6% on the canvas and 82.1% on the column;
the comma blocked from the answer token repeats in 72.4% to 83.3% (control 0.0%). On the canvas, gaze
heads sent to a written place: 0.0%, to a place that holds another shape: 37.7%; the shape token
blocked from the image never stops in 99.5%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import matching_letters, run_list_variant, variant_argument, variant_list_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model, n_layers
from utils.patching import Patch, make_patch, run_patches

SECOND_ANSWER_SOURCES = {"gaze_queries_next_panel": "source", "gaze_queries_written_panel": "source_written",
                         "gaze_queries_other_shape_panel": "source_other"}


def first_answer(model, pair, variant, results_row):
    """Nothing written. Patches from the source whose first matching panel is r2."""
    target, source = run_list_variant(model, pair["target"], variant, []), run_list_variant(model, pair["source_first"], variant, [])
    read_from = range(target.token("shape"), target.token("colon"))
    patches = []
    for binding, relay, gaze in ((HEADS["binding"], HEADS["relay"], HEADS["gaze"]),
                                 tuple(random_like(HEADS[name], seed=pair["index"]) for name in ("binding", "relay", "gaze"))):
        patches += [make_patch(source, target, "head_output", "shape", heads=binding),
                    make_patch(source, target, "values", "colon", heads=relay, positions=read_from),
                    make_patch(source, target, "query", "colon", heads=gaze)]
    moved = [answer == " " + pair["target"]["letters"][pair["panels"][1]] for answer in run_patches(model, target, patches)["answers"]]
    results_row.update(first_binding_outputs=moved[0], first_relay_values=moved[1], first_gaze_queries=moved[2],
                       first_random_heads=any(moved[3:]))


def second_answer(model, pair, variant, results_row):
    """With "Answer: W," written: gaze queries from sources whose next panel differs, and the blocks."""
    W, own = pair["written"], " " + matching_letters(pair["target"])[1]
    target = run_list_variant(model, pair["target"], variant, [W])
    comma, answer, colon, shape = (target.token(name) for name in ("last_comma", "last_answer", "colon", "shape"))
    every_layer = range(n_layers(model))
    random_gaze = random_like(HEADS["gaze"], seed=pair["index"])
    for name, side in SECOND_ANSWER_SOURCES.items():
        if side not in pair:
            continue
        source = run_list_variant(model, pair[side], variant, [W])
        panel = pair[side]["matching"][1]                                   # the panel of the source's next answer
        if source.answer != " " + pair[side]["letters"][panel]:
            continue
        there = " " + pair["target"]["letters"][panel]
        gaze, control = run_patches(model, target, [make_patch(source, target, "query", "last_comma", heads=heads)
                                                    for heads in (HEADS["gaze"], random_gaze)])["answers"]
        results_row.update({name: gaze == there, name + "_random_heads": control == there})
    patches = [Patch().block(None, [comma], [answer], layers=every_layer),
               Patch().block(None, [comma], [colon], layers=every_layer),
               Patch().block(HEADS["relay"], [comma], range(shape, comma)),
               Patch().block(None, [answer, comma], target.image_positions, layers=every_layer)]
    blocked, control, relay, blind = run_patches(model, target, patches)["answers"]
    results_row.update(comma_blocked_repeats=blocked == " " + W, colon_blocked_repeats=control == " " + W,
                       colon_blocked_unchanged=control == own, relay_heads_blocked_not_own=relay != own,
                       answer_and_comma_blind_not_own=blind != own)


def stopping(model, pair, variant, results_row):
    """The image whose list ends after W: block its shape token from the image and see if it goes on."""
    finished = run_list_variant(model, pair["source_stop"], variant, [pair["written"]], end_on_letter=True)
    patch = Patch().block(None, [finished.token("shape")], finished.image_positions, layers=range(n_layers(model)))
    results_row["shape_token_blind_never_stops"] = run_patches(model, finished, [patch])["answers"][0] == ","


def main():
    variant = variant_argument()
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    results = Results(f"list_task_on_other_images_{variant['name']}", args)
    for pair in variant_list_pairs(model, variant, args.split, args.start + args.num_pairs)[args.start:]:
        row = {}
        first_answer(model, pair, variant, row)
        second_answer(model, pair, variant, row)
        stopping(model, pair, variant, row)
        results.add(pair["index"], **row)
    for condition in ("first_binding_outputs", "first_relay_values", "first_gaze_queries", "first_random_heads",
                      "gaze_queries_next_panel", "gaze_queries_next_panel_random_heads", "gaze_queries_written_panel",
                      "gaze_queries_other_shape_panel", "comma_blocked_repeats", "colon_blocked_repeats", "colon_blocked_unchanged",
                      "relay_heads_blocked_not_own", "answer_and_comma_blind_not_own", "shape_token_blind_never_stops"):
        results.report(condition)
    results.save(variant=variant["name"])


if __name__ == "__main__":
    main()
