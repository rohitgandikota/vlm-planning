"""Zeroing the language heads, and setting a language, on the 2B and 4B (Appendix G.1.5, Table "sizes-language", rows 6 to 8).

No source run. Two tests on each size's own 16 language heads, on the colored-shapes task:
    zeroing          the outputs of the language heads at the last prompt token are set to zero. If
                     they add the language, German and French runs fall back to English. Chinese
                     runs are counted when they keep Chinese.
    language vector  for each language L, the mean over the 12 search images of what the language
                     heads together add to the residual stream at the last prompt token in L, minus
                     the same in English. We add it (times 0.5 for German and French, 1.0 for
                     Chinese) to the residual of an English run at the last prompt token, entering
                     layer 24 of 36 (scaled by depth on other sizes), and count runs that answer in L.
The edit stays on while the model writes its answer.

    python generalization/main_experiments/14_sizes_language_zeroing_and_vector.py --model Qwen/Qwen3-VL-2B-Instruct --num_pairs 500

--num_pairs is the number of held-out images (the paper's runs used 100). Models: Qwen3-VL-2B (16 GB)
or 4B (24 GB). 6 written answers of up to 8 tokens per image.

Paper: zeroing, German and French to English 76% (2B), 54% (4B), 94.0% (8B); Chinese kept 7%, 100%,
100.0%. Language vector to German / French / Chinese 75 / 75 / 100% (2B), 32 / 48 / 92% (4B).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import LANGUAGES, answer_language, color_images, run_color_question
from utils.experiment import Results, parse_args
from utils.heads import heads_of_size
from utils.model import load_model, n_layers
from utils.patching import Patch, write_with_fixed_patch, written_by_heads

SCALE = {"de": 0.5, "fr": 0.5, "zh": 1.0}
PAPER = {"zeroed_to_english": {"2B": 76, "4B": 54, "8B": 94.0}, "zeroed_chinese_kept": {"2B": 7, "4B": 100, "8B": 100.0},
         "vector_de": {"2B": 75, "4B": 32}, "vector_fr": {"2B": 75, "4B": 48}, "vector_zh": {"2B": 100, "4B": 92}}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    heads = heads_of_size(args.model, "language")
    layer = round(24 * n_layers(model) / 36)
    sums = {language: [] for language in LANGUAGES}                    # the language vectors, from the search images
    for spec in color_images("search"):
        for language in LANGUAGES:
            sums[language].append(written_by_heads(model, run_color_question(model, spec, language), heads, "last"))
    vectors = {language: (torch.stack(sums[language]) - torch.stack(sums["en"])).mean(0) for language in SCALE}
    results = Results(f"sizes_language_zeroing_and_vector_{args.model.split('/')[-1]}", args)
    for spec in color_images(args.split, args.start + args.num_pairs)[args.start:]:
        for language in ("de", "fr", "zh"):
            run = run_color_question(model, spec, language)
            zero = Patch()
            for l, h in heads:
                zero.head_output(l, h, run.last, torch.zeros_like(run.head_output(l, h, "last")))
            spoken = answer_language(write_with_fixed_patch(model, run, zero), spec["shape"])
            if language == "zh":
                results.add(spec["index"], zeroed_chinese_kept=spoken == "zh")
            else:
                results.add(spec["index"], zeroed_to_english=spoken == "en", language=language)
        english = run_color_question(model, spec, "en")
        for language, scale in SCALE.items():
            moved = Patch().residual(layer, english.last, english.residual(layer, "last") + scale * vectors[language])
            spoken = answer_language(write_with_fixed_patch(model, english, moved), spec["shape"])
            results.add(spec["index"], **{f"vector_{language}": spoken == language})
    size = args.model.split("-")[-2]
    results.report("zeroed_to_english", paper=PAPER["zeroed_to_english"].get(size), label="heads zeroed: German and French runs to English")
    results.report("zeroed_chinese_kept", paper=PAPER["zeroed_chinese_kept"].get(size), label="heads zeroed: Chinese runs kept Chinese")
    for language in SCALE:
        results.report(f"vector_{language}", paper=PAPER[f"vector_{language}"].get(size), label=f"language vector: English run to {language}")
    results.save()


if __name__ == "__main__":
    main()
