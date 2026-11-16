"""What the gaze heads return does not depend on the language (Section 6.2).

On the colored-shapes task we take what the 32 gaze heads together add to the residual stream at
the last prompt token: the sum of their outputs after the output projection. No patch. We compare
this vector by its cosine
    for one image asked in two languages (all 6 pairs of the four languages), and
    for two images asked in English (the image and each of its three sources, which show another
    cyan shape).
We also read the vector against the output vocabulary (LogitLens) and print in which language its
top token names the cyan shape.

    python language/main_experiments/02_gaze_heads_in_logit_lens.py --num_pairs 500

--num_pairs is the number of images.

Paper: cosine 0.95 for one image in two languages, and 0.50 for two images in one language. The
top token is English or Chinese, whatever the language of the question.
"""
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.patching import top_tokens, written_by_heads
from utils.dataset import LANGUAGES, answer_language, color_images, run_color_question
from utils.experiment import Results, interval, parse_args
from utils.heads import HEADS
from utils.model import load_model


def gaze_output(model, spec, language):
    return written_by_heads(model, run_color_question(model, spec, language), HEADS["gaze"], "last")


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def report_mean(results, name, paper):
    rows = [row for row in results.rows if row["compared"] == name]
    values, images = [row["cosine"] for row in rows], [row["pair"] for row in rows]
    low, high = interval(values, [1.0] * len(values), images)
    print(f"cosine, {name:28s} {sum(values) / len(values):.3f}  [{low / 100:.3f}, {high / 100:.3f}]  ({len(values)} values)   paper: {paper}")


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("gaze_heads_in_logit_lens", args)
    top_language = {language: Counter() for language in LANGUAGES}
    for spec in color_images(args.split, args.start + args.num_pairs)[args.start:]:
        vectors = {language: gaze_output(model, spec, language) for language in LANGUAGES}
        for i, a in enumerate(LANGUAGES):
            for b in LANGUAGES[i + 1:]:
                results.add(spec["index"], compared="one image, two languages", languages=f"{a}-{b}",
                            cosine=cosine(vectors[a], vectors[b]))
        for source in spec["sources"]:
            results.add(spec["index"], compared="two images, one language", source_image=source["index"],
                        cosine=cosine(vectors["en"], gaze_output(model, source, "en")))
        for language in LANGUAGES:
            top_language[language][answer_language(top_tokens(model, vectors[language], 1)[0], spec["shape"])] += 1
    report_mean(results, "one image, two languages", paper=0.95)
    report_mean(results, "two images, one language", paper=0.50)
    for language in LANGUAGES:
        print(f"question in {language}: the top token names the cyan shape in {dict(top_language[language])}")
    results.save(top_token_language={language: dict(count) for language, count in top_language.items()})


if __name__ == "__main__":
    main()
