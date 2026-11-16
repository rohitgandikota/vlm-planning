"""The skip experiment at every model size (Appendix G.1.4, Table "sizes-progress", first row).

The design of progress/main_experiments/02_progress_at_the_comma_token.py: the source has written
the letter of its first matching panel, which is also the target's first matching panel. The target
has written nothing. We copy the source's comma-token residual into the target's colon token, at one
layer at a time, every layer of the model. If the comma token holds "one panel done", the target
skips its first matching panel and gives its second answer.

    python generalization/main_experiments/07_sizes_skip_by_layer.py --model Qwen/Qwen3-VL-32B-Instruct --num_pairs 500

The pairs are drawn for this experiment and kept by the model's own no-patch check
(`progress_pairs_of`), so --split is not used. The paper's runs on the other sizes drew their pairs
from another seed with a source of three matching panels (200 pairs on the 2B and 4B, 100 on the 32B).
Models: any Qwen3-VL size. 2B 16 GB, 4B 24 GB, 8B 48 GB, 32B 160 GB (one B200) or --device cpu.
(number of layers + 1) patched runs per pair.

Paper: the second answer peaks at 78% at layer 17 of 28 (2B), 74% at 20 of 36 (4B), 91.0% at 20 of 36
(8B), 99% at 47 of 64 (32B). The 32B shows no early skip (10% or less through layer 40).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import name_answer, progress_pairs_of, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import make_patch, run_patches

OUTCOMES = ("own", "second", "pointer", "copy")
PAPER = {"2B": "78 at layer 17", "4B": "74 at layer 20", "8B": "91.0 at layer 20", "32B": "99 at layer 47"}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results(f"sizes_skip_by_layer_{args.model.split('/')[-1]}", args)
    layers = range(n_layers(model) + 1)                 # the last one is the input of the final norm
    for pair in progress_pairs_of(model, args.model, "progress", args.start + args.num_pairs)[args.start:]:
        source, target = run_list(model, pair["source"]), run_list(model, pair["target"])
        patches = [make_patch(source, target, "residual", "colon", source_token="last_comma", layer=layer) for layer in layers]
        outcomes = [name_answer(answer, pair["letters"]) for answer in run_patches(model, target, patches)["answers"]]
        results.add(pair["index"], outcomes=outcomes,
                    **{f"{name}_layer_{layer}": outcome == name for layer, outcome in zip(layers, outcomes) for name in OUTCOMES})
    print("layer  " + "  ".join(f"{name:>7s}" for name in OUTCOMES))
    for layer in layers:
        print(f"{layer:5d}  " + "  ".join(f"{results.rate(f'{name}_layer_{layer}')['percent']:7.1f}" for name in OUTCOMES))
    peak = max(layers, key=lambda layer: results.rate(f"second_layer_{layer}")["percent"])
    results.report(f"second_layer_{peak}", paper=PAPER.get(args.model.split("-")[-2]),
                   label=f"peak at layer {peak} of {n_layers(model)}: second answer")
    results.save()


if __name__ == "__main__":
    main()
