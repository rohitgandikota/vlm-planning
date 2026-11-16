"""Is the shape token read, in other model families (Appendix G.2, row 3 of Table "model-families").

No source run. In the target's own one-answer run, no later token may attend to the shape token, in
every head of every layer. If the later tokens must read the pointer or the request there, the
model loses its answer. The control blocks the attention to the token before the shape token ("the")
instead, which should cost nothing. Both sides of each kept pair are a target once.

    python generalization/main_experiments/17_families_shape_token_blocked.py --model google/gemma-3-12b-it --num_pairs 50

--num_pairs is the number of kept pairs (two cases each; the paper used 100 questions); --start is
the first candidate. Models: see 15_families_no_patch.py; FP32 on one GPU with 80 GB. 2 full forward
passes per case.

Paper, answers lost: Qwen3-VL-8B 77, LLaVA-OneVision-7B 70, InternVL3-8B 80, Gemma-3-12B 73, Pixtral-12B
75, Idefics3-8B 69. The control keeps every answer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.patching import OtherVLM, letter_of
from utils.dataset import family_one_answer_pairs, one_answer_image, one_answer_question
from utils.experiment import Results, parse_args

PAPER = {"Qwen3-VL-8B-Instruct": 77, "llava-onevision-qwen2-7b-ov-hf": 70, "InternVL3-8B-hf": 80, "gemma-3-12b-it": 73,
         "pixtral-12b": 75, "Idefics3-8B-Llama3": 69}


def main():
    args = parse_args(__doc__, num_pairs=50)
    vlm = OtherVLM(args.model, args.device)
    results = Results(f"families_shape_token_blocked_{args.model.split('/')[-1]}", args)
    for pair in family_one_answer_pairs(vlm, args.num_pairs, first=args.start):
        for side in (pair["source"], pair["target"]):
            prompt = vlm.prompt(one_answer_image(side), one_answer_question(side["ask"]), shape=side["ask"])
            shape, later = prompt.positions["shape"], range(prompt.positions["shape"] + 1, len(prompt.ids))
            own = side["letters"][side["shapes"].index(side["ask"])]
            blocked, _ = vlm.run(prompt, block=[(later, [shape])])
            control, _ = vlm.run(prompt, block=[(later, [shape - 1])])
            results.add(pair["index"], answer_lost=letter_of(blocked) != own, control_kept=letter_of(control) == own, answer=blocked)
    results.report("answer_lost", paper=PAPER.get(args.model.split("/")[-1]), label="later tokens cannot read the shape token: answer lost")
    results.report("control_kept", paper=100, label="cannot read the token before it: answer kept")
    results.save()


if __name__ == "__main__":
    main()
