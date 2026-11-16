"""How often each model size answers right with no patch (Appendix G.1.1).

The runs on the 2B, 4B and 32B use the 8B's pairs (`pairs_of_8b`), so a smaller model can get some
of them wrong. With no patch we count
    single answer   the next token after "Answer:" is the letter inside the asked shape, for each
                    side of each one-answer pair
    list            with the letter W written, both runs of a list pair (target and source) give
                    their second letter

    python generalization/main_experiments/01_sizes_no_patch.py --model Qwen/Qwen3-VL-2B-Instruct --num_pairs 500

Models: any Qwen3-VL size. 2B 16 GB, 4B 24 GB, 8B 48 GB, 32B 160 GB (one B200) or --device cpu.
No patched runs: 2 runs per one-answer pair and 2 per list pair.

Paper (an earlier, smaller run): single answer 93.4% (2B), 96.7% (4B), 98.9% (8B); list 48.0% (2B),
92.5% (4B), 100.0% (32B), 97.4% (8B).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import matching_letters, one_answer_image, one_answer_question, pairs_of_8b, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model
from utils.patching import run_model

PAPER = {"single_answer": {"2B": 93.4, "4B": 96.7, "8B": 98.9}, "list": {"2B": 48.0, "4B": 92.5, "8B": 97.4, "32B": 100.0}}


def main():
    args = parse_args(__doc__)
    one_answer = pairs_of_8b("one_answer", args.split, args.start + args.num_pairs, args.device)[args.start:]
    lists = pairs_of_8b("list", args.split, args.start + args.num_pairs, args.device)[args.start:]
    model = load_model(args.model, args.device)
    results = Results(f"sizes_no_patch_{args.model.split('/')[-1]}", args)
    for pair in one_answer:
        for side in (pair["source"], pair["target"]):
            answer = run_model(model, one_answer_image(side), one_answer_question(side["ask"]), record=False).answer
            results.add(pair["index"], single_answer=answer == " " + side["letters"][side["shapes"].index(side["ask"])])
    for pair in lists:
        written = [pair["written"]]
        right = [run_list(model, pair[side], written, record=False).answer == " " + matching_letters(pair[side])[1]
                 for side in ("target", "source")]
        results.add(pair["index"], list=all(right))
    size = args.model.split("-")[-2]                                 # "2B", "4B", "8B" or "32B"
    results.report("single_answer", paper=PAPER["single_answer"].get(size), label="single answer right")
    results.report("list", paper=PAPER["list"].get(size), label="list: both runs right on the second letter")
    results.save()


if __name__ == "__main__":
    main()
