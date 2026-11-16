"""Progress at the comma token (Section 5.1, Figure 3b).

Where does the model keep track of how far it has come? The source has written one letter, from
the panel that is also the target's first matching panel. The target has written nothing. We patch
the residual vector of the source's comma token into the target's colon token, one layer at a time.
The two runs ask for different shapes and share no letter, so the target can answer with

    own       its own first answer (nothing changed)
    second    its second answer: the patch carried progress past the first panel
    pointer   its letter at the source's next panel: the patch carried the source's next pointer
    copy      the source's next letter

    python progress/main_experiments/02_progress_at_the_comma_token.py --num_pairs 500

Paper: the target gives its second answer in 91.0% of pairs at layer 20 and in 70.0% at layer 21.
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
    results = Results("progress_at_the_comma_token", args)
    layers = range(n_layers(model) + 1)                 # the last one is the input of the final norm
    for pair in progress_pairs(model, "progress", args.start + args.num_pairs)[args.start:]:
        source, target = run_list(model, pair["source"]), run_list(model, pair["target"])
        patches = [make_patch(source, target, "residual", "colon", source_token="last_comma", layer=layer)
                   for layer in layers]
        answers = run_patches(model, target, patches)["answers"]
        outcomes = [name_answer(answer, pair["letters"]) for answer in answers]       # one per layer
        results.add(pair["index"], outcomes=outcomes, shared_panel=pair["shared_panel"],    # Appendix D.2 splits by it
                    **{f"layer_{layer}": outcome == "second" for layer, outcome in zip(layers, outcomes)})
    results.report("layer_20", paper=91.0, label="layer 20: target gives its second answer")
    results.report("layer_21", paper=70.0, label="layer 21: target gives its second answer")
    results.save()


if __name__ == "__main__":
    main()
