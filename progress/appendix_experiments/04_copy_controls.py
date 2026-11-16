"""The copy itself does nothing, and the state needs a real letter (Appendix D.2, last paragraph).

Two controls for progress/main_experiments/02_progress_at_the_comma_token.py, on pairs of the same
design: the source has written its letter from the panel that is also the target's first matching
panel (panel 1, 2 or 3), and the target has written nothing. We patch the residual vector of the
source's last token into the target's colon token, one layer at a time, for three sources:
    real      the source as in the main experiment (its comma token after one written letter)
    nothing   the same source image with nothing written (its colon token)
    absent    the same source image prefilled with a letter that is drawn in neither image
The target skips its first matching panel when it gives its second answer.

    python progress/appendix_experiments/04_copy_controls.py --num_pairs 500

Paper: with nothing written, the target keeps its own answer in 98.6 to 100.0% of pairs at every
layer through 29. With the absent letter, it skips in 53.8% of pairs at layer 20, against 91.0% for
the real source (91.0% is the main experiment's value; on these pairs the real source gives 91.2%).
By the shared panel it skips in 96.8% (panel 1), 27.6% (panel 2), and 29.0% (panel 3).
These pairs are drawn for this experiment (utils.dataset.control_pairs), so --split is not used.
Here the source has its shape at the shared panel and at two random later panels.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import control_pairs, name_answer, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("copy_controls", args)
    layers = range(n_layers(model) + 1)
    for pair in control_pairs(model, args.start + args.num_pairs)[args.start:]:
        target = run_list(model, pair["target"])
        sources = {"real": run_list(model, pair["source"]),
                   "nothing": run_list(model, pair["source"], []),
                   "absent": run_list(model, pair["source"], [pair["absent"]])}
        row = {"shared_panel": pair["shared_panel"]}
        for name, source in sources.items():
            patches = [make_patch(source, target, "residual", "colon", source_token="last", layer=layer) for layer in layers]
            said = [name_answer(answer, pair["letters"]) for answer in run_patches(model, target, patches)["answers"]]
            row[name] = said                                              # saved with the row, not counted
            row.update({f"{name}_layer_{layer}_own": s == "own" for layer, s in zip(layers, said)})
            row.update({f"{name}_layer_{layer}_second": s == "second" for layer, s in zip(layers, said)})
        for panel in range(3):                                            # the split by the shared panel
            for name in ("real", "absent"):
                row[f"{name}_panel_{panel + 1}_layer_20_second"] = row[f"{name}_layer_20_second"] if panel == pair["shared_panel"] else None
        results.add(pair["index"], **row)
    own = [results.rate(f"nothing_layer_{layer}_own")["percent"] for layer in range(30)]
    if results.rows:
        print(f"nothing written: own answer kept in {min(own):.1f} to {max(own):.1f}% at layers 0 to 29   paper: 98.6 to 100.0")
    results.report("absent_layer_20_second", paper=53.8, label="absent letter, layer 20: second answer")
    results.report("real_layer_20_second", paper="91.0 (91.2 on these pairs)", label="real source, layer 20: second answer")
    for panel, paper in zip((1, 2, 3), (96.8, 27.6, 29.0)):
        results.report(f"absent_panel_{panel}_layer_20_second", paper=paper, label=f"absent letter, shared panel {panel}: second answer")
    results.save()


if __name__ == "__main__":
    main()
