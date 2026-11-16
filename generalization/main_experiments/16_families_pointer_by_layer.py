"""Does one residual vector carry the pointer, in other model families (Appendix G.2, rows 1 and 2 of Table "model-families").

The residual patch of Section 4 on any model: we copy the source's residual at one token into the
target's run at the same token, one layer at a time, every layer, in both directions of each
one-answer pair. A model whose residual holds the pointer answers K, its own letter at the panel
the source points to.
    row 1   at the colon token
    row 2   at the shape token (early layers move the request instead: outcome P)
The controls copy a token that cannot hold the pointer: the source's token before the colon token
("Answer") into the target's colon token, and the source's token before the shape token ("the")
into the target's shape token. A pair is kept when the model answers both questions right with no
patch (`family_one_answer_pairs`). We report the layer where K peaks.

    python generalization/main_experiments/16_families_pointer_by_layer.py --model mistral-community/pixtral-12b --num_pairs 100

--num_pairs is the number of kept pairs (the paper used 100); --start is the first candidate.
Models: see 15_families_no_patch.py; FP32 on one GPU with 80 GB. 8 x (number of layers + 1) full
forward passes per pair.

Paper (K at the peak layer, colon / shape token): Qwen3-VL-8B 80.0 (24 of 36) / 42.5 (20);
LLaVA-OneVision-7B 84.0 (22 of 28) / 69.5 (17); InternVL3-8B 65.5 (22 of 28) / 68.0 (17); Gemma-3-12B
30.5 (34 of 48) / 32.0 (20); Pixtral-12B 68.0 (25 of 40) / 79.0 (16); Idefics3-8B 38.5 (17 of 32) /
31.0 (12). The colon control gives K in 0.0% in every model; the shape-token control at most chance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.patching import OtherVLM, name_letter
from utils.dataset import family_one_answer_pairs, one_answer_image, one_answer_question, outcome_letters
from utils.experiment import Results, parse_args

PAPER = {"Qwen3-VL-8B-Instruct": (80.0, 42.5), "llava-onevision-qwen2-7b-ov-hf": (84.0, 69.5), "InternVL3-8B-hf": (65.5, 68.0),
         "gemma-3-12b-it": (30.5, 32.0), "pixtral-12b": (68.0, 79.0), "Idefics3-8B-Llama3": (38.5, 31.0)}


def main():
    args = parse_args(__doc__, num_pairs=100)
    vlm = OtherVLM(args.model, args.device)
    layers = range(vlm.n_layers + 1)
    results = Results(f"families_pointer_by_layer_{args.model.split('/')[-1]}", args)
    for pair in family_one_answer_pairs(vlm, args.num_pairs, first=args.start):
        prompts, states = {}, {}
        for name in ("source", "target"):
            side = pair[name]
            prompts[name] = vlm.prompt(one_answer_image(side), one_answer_question(side["ask"]), shape=side["ask"])
            colon, shape = prompts[name].positions["colon"], prompts[name].positions["shape"]
            states[name] = vlm.run(prompts[name], record=[colon, colon - 1, shape, shape - 1])[1]
        for source, target in (("source", "target"), ("target", "source")):
            letters = outcome_letters(pair[source], pair[target])
            row = {}
            for token in ("colon", "shape"):
                here, there = prompts[target].positions[token], prompts[source].positions[token]
                main_run = vlm.sweep(prompts[target], here, states[source][there])
                control = vlm.sweep(prompts[target], here, states[source][there - 1])      # the token before it
                for layer, answer, answer_control in zip(layers, main_run, control):
                    outcome = name_letter(answer, letters)
                    row[f"{token}_K_layer_{layer}"] = outcome == "K"
                    row[f"{token}_P_layer_{layer}"] = outcome == "P"
                    row[f"{token}_control_K_layer_{layer}"] = name_letter(answer_control, letters) == "K"
            results.add(pair["index"], **row)
    for token, row_number in (("colon", 1), ("shape", 2)):
        rate = lambda name, layer: results.rate(f"{token}_{name}_layer_{layer}")["percent"]
        peak = max(layers, key=lambda layer: rate("K", layer))
        paper = PAPER.get(args.model.split("/")[-1], (None, None))[row_number - 1]
        results.report(f"{token}_K_layer_{peak}", paper=paper, label=f"row {row_number}, {token} token: K at layer {peak} of {vlm.n_layers}")
        print(f"   control at the {token} token: K in at most {max(rate('control_K', layer) for layer in layers):.1f}% at any layer;"
              f" P peaks at {max(rate('P', layer) for layer in layers):.1f}%")
    results.save()


if __name__ == "__main__":
    main()
