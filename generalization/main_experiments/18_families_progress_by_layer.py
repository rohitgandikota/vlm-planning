"""Does the comma token hold progress, in other model families (Appendix G.2, rows 4 and 5 of Table "model-families").

The two residual experiments of Sections 5.1 and 5.2 on any model. We copy the source's residual at
its last comma token into the target's colon token, one layer at a time, every layer.
    row 4, skip     the design of progress/main_experiments/02: the source has written the letter
                    of the panel it shares with the target. The target should skip that panel and
                    give its second answer.
    row 5, cursor   the design of progress/main_experiments/03: the source has written panels 2
                    and 4. The cursor reading answers panel 6 ("position"); the set reading answers
                    panel 1 ("set"), which we report at the layer where the cursor peaks.
The control copies the source's colon token, from the same source before it wrote anything. A pair
is kept when its no-patch runs give the expected next letters (`family_progress_pairs`).

    python generalization/main_experiments/18_families_progress_by_layer.py --model OpenGVLab/InternVL3-8B-hf --num_pairs 100

--num_pairs is the number of kept pairs of each experiment (the paper used 100). Run it only on
models that answer at least 70% of the lists right (15_families_no_patch.py). Models: see
15_families_no_patch.py; FP32 on one GPU with 80 GB. 4 x (number of layers + 1) full forward passes
per pair.

Paper (peak percent, peak layer of the number of layers): skip, Qwen3-VL-8B 96 (20 of 36), InternVL3-8B
79 (19 of 28), Gemma-3-12B 96 (23 of 48), Pixtral-12B 87 (18 of 40); cursor (set at that layer) 99 (0),
94 (0), 86 (0), 89 (2) at layers 21, 19, 26, 18. The control gives the skip in at most 3% and the
cursor in 0%. LLaVA-OneVision and Idefics3 do not solve the list task.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.patching import OtherVLM, name_letter
from utils.dataset import family_progress_pairs, list_image, list_question
from utils.experiment import Results, parse_args

PAPER = {"Qwen3-VL-8B-Instruct": (96, 99), "InternVL3-8B-hf": (79, 94), "gemma-3-12b-it": (96, 86), "pixtral-12b": (87, 89)}
EXPECTED = {"progress": "second", "position": "position"}       # the outcome of the copied state in each experiment


def main():
    args = parse_args(__doc__, num_pairs=100)
    vlm = OtherVLM(args.model, args.device)
    layers = range(vlm.n_layers + 1)
    results = {experiment: Results(f"families_{experiment}_by_layer_{args.model.split('/')[-1]}", args) for experiment in EXPECTED}
    for experiment, expected in EXPECTED.items():
        for pair in family_progress_pairs(vlm, experiment, args.num_pairs):
            source, target = pair["source"], pair["target"]
            question = list_question(source["ask"])
            written = vlm.prompt(list_image(source), question, source["written"])
            nothing = vlm.prompt(list_image(source), question)
            comma, colon = written.positions["last_comma"], nothing.positions["colon"]
            target_prompt = vlm.prompt(list_image(target), list_question(target["ask"]))
            main_run = vlm.sweep(target_prompt, target_prompt.positions["colon"], vlm.run(written, record=[comma])[1][comma])
            control = vlm.sweep(target_prompt, target_prompt.positions["colon"], vlm.run(nothing, record=[colon])[1][colon])
            row = {}
            for layer, answer, answer_control in zip(layers, main_run, control):
                row[f"layer_{layer}"] = name_letter(answer, pair["letters"]) == expected
                row[f"set_layer_{layer}"] = name_letter(answer, pair["letters"]) == "set"
                row[f"control_layer_{layer}"] = name_letter(answer_control, pair["letters"]) == expected
            results[experiment].add(pair["index"], **row)
    paper = PAPER.get(args.model.split("/")[-1], (None, None))
    for (experiment, expected), paper_value in zip(EXPECTED.items(), paper):
        result = results[experiment]
        peak = max(layers, key=lambda layer: result.rate(f"layer_{layer}")["percent"])
        result.report(f"layer_{peak}", paper=paper_value, label=f"{experiment}: {expected} at the peak layer {peak} of {vlm.n_layers}")
        if experiment == "position":
            result.report(f"set_layer_{peak}", label="position: set at the same layer")
        print(f"   control: {expected} in at most {max(result.rate(f'control_layer_{layer}')['percent'] for layer in layers):.1f}% at any layer")
        result.save()


if __name__ == "__main__":
    main()
