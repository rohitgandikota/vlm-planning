"""The comma token and select, in other model families (Appendix G.2, rows 6 and 7 of Table "model-families").

Two blocks in the model's own run of a plain list question (the asked shape in 3 or 4 panels):
    row 6   one letter written, "Answer: W,". The comma token cannot attend to the answer token, in
            every head of every layer. Without the cursor there, the model should write W again.
            Control: the comma token cannot attend to the colon token.
    row 7   nothing written. The image tokens of the first matching panel are hidden from every
            text token, at every layer. If select falls back, the model answers with the second
            matching panel. Control: a panel without the asked shape is hidden instead.
A question is kept when the model gives the right next letter with no patch. The image tokens of
each panel are found by experiment (`OtherVLM.image_token_panels`), since other models cut the image
into tiles.

    python generalization/main_experiments/19_families_comma_and_hidden_panel.py --model google/gemma-3-12b-it --num_pairs 100

--num_pairs is the number of kept questions of each test (the paper used 100). Run it only on models
that answer at least 70% of the lists right (15_families_no_patch.py). Models: see
15_families_no_patch.py; FP32 on one GPU with 80 GB. Row 6: 3 full forward passes per question;
row 7: 10.

Paper: repeats, Qwen3-VL-8B 82, InternVL3-8B 56, Gemma-3-12B 97, Pixtral-12B 90 (control: no
repeats); falls back to the second matching panel 48, 70, 88, 21 (control: at most 1% move).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.patching import OtherVLM, letter_of
from utils.dataset import list_image, list_question, sample_list_question
from utils.experiment import Results, parse_args

PAPER = {"Qwen3-VL-8B-Instruct": (82, 48), "InternVL3-8B-hf": (56, 70), "gemma-3-12b-it": (97, 88), "pixtral-12b": (90, 21)}


def comma_blocked(vlm, args, results):
    """Row 6: the comma token cannot attend to the answer token."""
    rng, kept, index = random.Random(33), 0, -1
    while kept < args.num_pairs:
        index += 1
        side = sample_list_question(rng, matches=rng.choice([3, 4]))
        first, second = side["expected"][:2]
        prompt = vlm.prompt(list_image(side), list_question(side["ask"]), written=[first])
        if letter_of(vlm.run(prompt)[0]) != second:
            continue
        comma = [prompt.positions["last_comma"]]
        blocked, _ = vlm.run(prompt, block=[(comma, [prompt.positions["last_answer"]])])
        control, _ = vlm.run(prompt, block=[(comma, [prompt.positions["colon"]])])
        results.add(index, repeat=letter_of(blocked) == first, control_repeat=letter_of(control) == first)
        kept += 1


def panel_hidden(vlm, args, results):
    """Row 7: the first matching panel is hidden from every text token."""
    rng, kept, index = random.Random(66), 0, -1
    while kept < args.num_pairs:
        index += 1
        side = sample_list_question(rng, matches=rng.choice([3, 4]))
        image, question = list_image(side), list_question(side["ask"])
        prompt = vlm.prompt(image, question)
        if letter_of(vlm.run(prompt)[0]) != side["expected"][0]:
            continue
        panels = vlm.image_token_panels(image, question, image.regions)
        image_tokens = prompt.positions["image"]
        text = range(image_tokens[-1] + 1, len(prompt.ids))
        tokens_of = lambda panel: [position for position, p in zip(image_tokens, panels) if p == panel]
        filler = next(panel for panel in range(6) if panel not in side["matching"])
        hidden, _ = vlm.run(prompt, block=[(text, tokens_of(side["matching"][0]))])
        control, _ = vlm.run(prompt, block=[(text, tokens_of(filler))])
        results.add(index, falls_back=letter_of(hidden) == side["expected"][1], control_moves=letter_of(control) == side["expected"][1])
        kept += 1


def main():
    args = parse_args(__doc__, num_pairs=100)
    vlm = OtherVLM(args.model, args.device)
    paper = PAPER.get(args.model.split("/")[-1], (None, None))
    results = Results(f"families_comma_blocked_{args.model.split('/')[-1]}", args)
    comma_blocked(vlm, args, results)
    results.report("repeat", paper=paper[0], label="row 6, comma cannot read the answer token: repeat")
    results.report("control_repeat", paper=0, label="comma cannot read the colon token: repeat")
    results.save()
    results = Results(f"families_panel_hidden_{args.model.split('/')[-1]}", args)
    panel_hidden(vlm, args, results)
    results.report("falls_back", paper=paper[1], label="row 7, first matching panel hidden: second panel")
    results.report("control_moves", paper="at most 1", label="a non-matching panel hidden: second panel")
    results.save()


if __name__ == "__main__":
    main()
