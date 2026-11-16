"""Gaze heads return a concept, not a word (Section 6.2).

On the colored-shapes task, a source asks in one language about one image, and a target asks in
another language about another image with another cyan shape. While the target writes its answer,
we copy the outputs of the 32 gaze heads from the source, at the last prompt token and at every
written token. If the gaze heads carry the concept and not the word, the target should name the
source's shape in its own language.

    python language/main_experiments/03_gaze_heads_another_language.py --num_pairs 500

--num_pairs is the number of target images. Each has 3 sources and 9 pairs of languages (targets
in German, French, Chinese): 27 cases per image, 13,500 in the paper. As a control, 32 random heads
of any layer are patched in the same way (drawn once per target image).

Paper: the target names the source's shape in its own language in 82.0% of cases, and never
changes its language. Random heads: 0.0%.
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
    results = Results("gaze_heads_another_language", args)
    for spec in color_images(args.split, args.start + args.num_pairs)[args.start:]:
        order = np.random.default_rng(11000 + spec["index"]).permutation(len(every_head))
        random_heads = [every_head[i] for i in order[:32]]
        for source, target, case in color_cases(model, spec):
            text = write_tokens(model, target, source, HEADS["gaze"])
            gaze = patched_answer(text, case)
            control = patched_answer(write_tokens(model, target, source, random_heads), case)
            results.add(spec["index"], gaze_heads=gaze == "concept", language_changed=gaze == "language",
                        random_heads=control == "concept", outcome=gaze, outcome_random=control, text=text, **case)
    results.report("gaze_heads", paper=82.0, label="gaze heads: source's shape, target's language")
    results.report("language_changed", paper=0.0, label="gaze heads: the language changed")
    results.report("random_heads", paper=0.0, label="random heads: source's shape, target's language")
    results.save()


if __name__ == "__main__":
    main()
