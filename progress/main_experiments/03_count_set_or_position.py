"""Count, set, or position (Section 5.2, Figure 3c).

What form does progress take? The source has its shape at panels 2, 4 and 5 and has written two
letters, from panels 2 and 4. The target has its shape at panels 1, 2, 3 and 6 and has written
nothing. We patch the source's comma-token residual into the target's colon token, one layer at a
time, as in the previous experiment. Each form of progress predicts another answer of the target:

    count      "two answers done": the target's third match (panel 3)
    set        "panels 2 and 4 done": the target's first match (panel 1), its own answer
    position   "done up to panel 4": the target's first match after panel 4 (panel 6)

    python progress/main_experiments/03_count_set_or_position.py --num_pairs 500

Paper: count in 77.8 to 90.8% of pairs at layers 13 to 18. Position in 98.4% at layer 20 and 97.6%
at layer 21. Set in at most 1.2% at these two layers.
These pairs are drawn for this experiment (utils.dataset.progress_pairs), so --split is not used.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import name_answer, progress_pairs, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("count_set_or_position", args)
    layers = range(n_layers(model) + 1)                 # the last one is the input of the final norm
    for pair in progress_pairs(model, "position", args.start + args.num_pairs)[args.start:]:
        source, target = run_list(model, pair["source"]), run_list(model, pair["target"])
        patches = [make_patch(source, target, "residual", "colon", source_token="last_comma", layer=layer)
                   for layer in layers]
        answers = run_patches(model, target, patches)["answers"]
        outcomes = [name_answer(answer, pair["letters"]) for answer in answers]       # one per layer
        results.add(pair["index"], outcomes=outcomes,
                    **{f"{form}_layer_{layer}": outcome == form
                       for layer, outcome in zip(layers, outcomes) for form in ("count", "set", "position")})
    for layer in range(13, 19):
        results.report(f"count_layer_{layer}", paper="77.8 to 90.8", label=f"layer {layer}: count")
    results.report("position_layer_20", paper=98.4, label="layer 20: position")
    results.report("position_layer_21", paper=97.6, label="layer 21: position")
    results.report("set_layer_20", paper="at most 1.2", label="layer 20: set")
    results.report("set_layer_21", paper="at most 1.2", label="layer 21: set")
    results.save()


if __name__ == "__main__":
    main()
