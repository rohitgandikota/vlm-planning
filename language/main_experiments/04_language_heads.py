"""Language heads add the language of the question (Section 6.2).

On the colored-shapes task, a source asks in one language about one image, and a target asks in
another language about another image. While the target writes its answer, we copy the outputs of
the 16 language heads from the source, at the last prompt token and at every written token. If
these heads carry the language and not the concept, the target should name its own shape in the
source's language.

    python language/main_experiments/04_language_heads.py --num_pairs 500

--num_pairs is the number of target images. Each has 3 sources and 9 pairs of languages (targets
in German, French, Chinese): 27 cases per image, 13,500 in the paper. The control of the paper is
one fixed set of 16 random heads of any layer, patched at the last prompt token only.

Paper: the language moves in 95.3% of cases, and the source's shape is never copied. Random
heads: 0.0%.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.patching import write_tokens
from utils.dataset import color_cases, color_images, patched_answer
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import all_heads, load_model


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    every_head = all_heads(model)
    random_heads = [every_head[i] for i in np.random.default_rng(1016).choice(len(every_head), 16, replace=False)]
    results = Results("language_heads", args)
    for spec in color_images(args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, case in color_cases(model, spec):
            text = write_tokens(model, target, source, HEADS["language"])
            language = patched_answer(text, case)
            control = patched_answer(write_tokens(model, target, source, random_heads, every_token=False), case)
            results.add(spec["index"], language_heads=language == "language", shape_copied=language in ("word", "concept"),
                        random_heads=control == "language", outcome=language, outcome_random=control, text=text, **case)
    results.report("language_heads", paper=95.3, label="language heads: target's shape, source's language")
    results.report("shape_copied", paper=0.0, label="language heads: the source's shape was copied")
    results.report("random_heads", paper=0.0, label="random heads: target's shape, source's language")
    results.save()


if __name__ == "__main__":
    main()
