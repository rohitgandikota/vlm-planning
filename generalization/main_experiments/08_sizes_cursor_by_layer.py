"""The cursor experiment at every model size (Appendix G.1.4, Table "sizes-progress", rows 2 to 6).

The design of progress/main_experiments/03_count_set_or_position.py: the source has its shape at
panels 2, 4 and 5 and has written the letters of panels 2 and 4. The target has its shape at panels
1, 2, 3 and 6 and has written nothing. We copy the source's comma-token residual into the target's
colon token, one layer at a time, every layer of the model. Each reading of the copied state gives
another answer of the target:
    one_skipped   "one letter written": its second match (panel 2)
    count         "two letters written": its third match (panel 3)
    panel_4       the visited panel itself
    position      the first matching panel after the visited panel (panel 6): the cursor
    copy          the source's own next letter
    set           "panels 2 and 4 done": its first match (panel 1)

    python generalization/main_experiments/08_sizes_cursor_by_layer.py --model Qwen/Qwen3-VL-32B-Instruct --num_pairs 500

The pairs are drawn for this experiment and kept by the model's own no-patch check
(`progress_pairs_of`), so --split is not used. The paper's runs used 200 pairs on the 2B and 4B and
100 on the 32B.
Models: any Qwen3-VL size. 2B 16 GB, 4B 24 GB, 8B 48 GB, 32B 160 GB (one B200) or --device cpu.
(number of layers + 1) patched runs per pair.

Paper (percent of pairs, peak layer): one written 87 to 99 (3 to 7) on the 2B, 97 (7) on the 4B, 75 to
91 (34 to 40) on the 32B; count 100 (8 to 13), 65.5 (20), 90.8 (15 on the 8B), 93 (42); visited panel
41 (18), 49 (22), at most 1.6 (8B), at most 1 (32B); cursor 22 (20), 53 (23), 98.4 (20 on the 8B),
100 (48 of 64); the source's letter from layer 22, 30, 30 (8B), 53.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import name_answer, progress_pairs_of, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import make_patch, run_patches

OUTCOMES = ("set", "one_skipped", "count", "panel_4", "position", "copy")


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results(f"sizes_cursor_by_layer_{args.model.split('/')[-1]}", args)
    layers = range(n_layers(model) + 1)                 # the last one is the input of the final norm
    for pair in progress_pairs_of(model, args.model, "position", args.start + args.num_pairs)[args.start:]:
        source, target = run_list(model, pair["source"]), run_list(model, pair["target"])
        patches = [make_patch(source, target, "residual", "colon", source_token="last_comma", layer=layer) for layer in layers]
        outcomes = [name_answer(answer, pair["letters"]) for answer in run_patches(model, target, patches)["answers"]]
        results.add(pair["index"], outcomes=outcomes,
                    **{f"{name}_layer_{layer}": outcome == name for layer, outcome in zip(layers, outcomes) for name in OUTCOMES})
    print("layer  " + "  ".join(f"{name:>11s}" for name in OUTCOMES))
    for layer in layers:
        print(f"{layer:5d}  " + "  ".join(f"{results.rate(f'{name}_layer_{layer}')['percent']:11.1f}" for name in OUTCOMES))
    for name in ("count", "panel_4", "position"):
        peak = max(layers, key=lambda layer: results.rate(f"{name}_layer_{layer}")["percent"])
        results.report(f"{name}_layer_{peak}", label=f"{name}: peak at layer {peak} of {n_layers(model)}")
        results.report(f"set_layer_{peak}", label="set, at the same layer")
    results.save()


if __name__ == "__main__":
    main()
