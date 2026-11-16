"""How often a model of another family answers right with no patch (Appendix G.2, Table "model-families", first row).

Each model sees the paper's images and prompts through its own chat template, with the assistant
turn prefilled with "Answer:". With no patch we count
    single answer   the next token is the letter inside the asked shape (one side of each of the
                    first one-answer candidates)
    list            the model writes the whole list right (letters case-folded), for plain list
                    questions with the asked shape in 2 to 4 of the six panels
We run the list tests (files 18 and 19) only on models that get at least 70% of the lists right.

    python generalization/main_experiments/15_families_no_patch.py --model google/gemma-3-12b-it --num_pairs 200

--num_pairs is the number of questions of each kind (the paper used 200). The models (FAMILY_MODELS):
Qwen/Qwen3-VL-8B-Instruct, llava-hf/llava-onevision-qwen2-7b-ov-hf, OpenGVLab/InternVL3-8B-hf,
google/gemma-3-12b-it, mistral-community/pixtral-12b, HuggingFaceM4/Idefics3-8B-Llama3. Each runs in
FP32 on one GPU with 80 GB (the paper used one 96 GB card). No patched runs.

Paper (single answer / list): Qwen3-VL-8B 98.5 / 96.5; LLaVA-OneVision-7B 99.5 / 18.5; InternVL3-8B
99.5 / 80.0; Gemma-3-12B 100.0 / 99.0; Pixtral-12B 100.0 / 98.0; Idefics3-8B 88.5 / 3.5.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.patching import OtherVLM, letter_of
from utils.dataset import list_image, list_question, one_answer_image, one_answer_question, sample_list_question, sample_one_answer_pair
from utils.experiment import Results, parse_args

PAPER = {"Qwen3-VL-8B-Instruct": (98.5, 96.5), "llava-onevision-qwen2-7b-ov-hf": (99.5, 18.5), "InternVL3-8B-hf": (99.5, 80.0),
         "gemma-3-12b-it": (100.0, 99.0), "pixtral-12b": (100.0, 98.0), "Idefics3-8B-Llama3": (88.5, 3.5)}


def main():
    args = parse_args(__doc__, num_pairs=200)
    vlm = OtherVLM(args.model, args.device)
    results = Results(f"families_no_patch_{args.model.split('/')[-1]}", args)
    for candidate in range(args.start, args.start + args.num_pairs):
        side = sample_one_answer_pair(candidate)["target"]
        answer, _ = vlm.run(vlm.prompt(one_answer_image(side), one_answer_question(side["ask"])))
        results.add(candidate, single_answer=letter_of(answer) == side["letters"][side["shapes"].index(side["ask"])])
    rng = random.Random(11)
    for index in range(args.num_pairs):
        side = sample_list_question(rng)
        text = vlm.generate(vlm.prompt(list_image(side), list_question(side["ask"])))
        written = [item.strip().upper() for item in text.split(",") if item.strip()]
        results.add(index, list=written == side["expected"], text=text)
    paper = PAPER.get(args.model.split("/")[-1], (None, None))
    results.report("single_answer", paper=paper[0], label="single answer right")
    results.report("list", paper=paper[1], label="whole list right")
    results.save()


if __name__ == "__main__":
    main()
