"""The colored-shapes task on the 2B and 4B, with no patch (Appendix G.1.5, Table "sizes-language", rows 1 and 2).

Two measurements with no patch, as in language/main_experiments/01 and 02:
    the model names the cyan shape in the language of the question (English, German, French,
    Chinese), with nothing prefilled
    what the size's gaze heads together add to the residual stream at the last prompt token: its
    cosine between one image asked in two languages, and between two images asked in English
If the gaze heads return a concept and not a word, the first cosine is high and the second lower.

    python generalization/main_experiments/12_sizes_language_no_patch.py --model Qwen/Qwen3-VL-2B-Instruct --num_pairs 500

--num_pairs is the number of images (the paper's runs used 100 held-out images). The paper did not
run the language tasks on the 32B. Models: Qwen3-VL-2B (16 GB) or 4B (24 GB); the 8B gives the main
paper's numbers. No patched runs: 4 written answers and 7 recorded runs per image.

Paper: no patch 97.5% (2B), 98.2% (4B), 100.0% (8B); cosine across languages / across images 0.87 /
0.78 (2B), 0.88 / 0.71 (4B), 0.95 / 0.50 (8B).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import LANGUAGES, answer_language, color_images, run_color_question
from utils.experiment import Results, parse_args
from utils.heads import heads_of_size
from utils.model import load_model
from utils.patching import write_tokens, written_by_heads

PAPER = {"no_patch": {"2B": 97.5, "4B": 98.2, "8B": 100.0},
         "one image, two languages": {"2B": 0.87, "4B": 0.88, "8B": 0.95},
         "two images, one language": {"2B": 0.78, "4B": 0.71, "8B": 0.50}}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    gaze = heads_of_size(args.model, "gaze")
    size = args.model.split("-")[-2]
    results = Results(f"sizes_language_no_patch_{args.model.split('/')[-1]}", args)
    cosine = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    cosines = {"one image, two languages": [], "two images, one language": []}
    for spec in color_images(args.split, args.start + args.num_pairs)[args.start:]:
        vectors = {}
        for language in LANGUAGES:
            run = run_color_question(model, spec, language)
            vectors[language] = written_by_heads(model, run, gaze, "last")
            text = write_tokens(model, run)
            results.add(spec["index"], no_patch=answer_language(text, spec["shape"]) == language, language=language, text=text)
        cosines["one image, two languages"] += [cosine(vectors[a], vectors[b]) for i, a in enumerate(LANGUAGES) for b in LANGUAGES[i + 1:]]
        for source in spec["sources"]:
            other = written_by_heads(model, run_color_question(model, source, "en"), gaze, "last")
            cosines["two images, one language"].append(cosine(vectors["en"], other))
    results.report("no_patch", paper=PAPER["no_patch"].get(size), label="right shape in the question's language")
    for name, values in cosines.items():
        print(f"gaze heads' summed output, cosine, {name}: {sum(values) / len(values):.2f}   paper: {PAPER[name].get(size)}")
    results.save(cosines=cosines)


if __name__ == "__main__":
    main()
