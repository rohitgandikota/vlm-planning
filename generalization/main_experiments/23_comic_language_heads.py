"""Language heads on comic images (Appendix G.3.1, "Language heads on comic images").

The colored-shapes task of Section 6.2 with the four colored shapes drawn over a comic strip (one
comic per image, under and around the shapes). As in language/main_experiments/04, a source asks in
one language about one image and a target asks in another language about another image. While the
target writes, we copy the outputs of the 16 language heads selected on these comic images
(HEADS["comic_language"]) from the source, at the last prompt token and at every written token. If
these heads carry the language, the target names its own shape in the source's language.

    python generalization/main_experiments/23_comic_language_heads.py --comics /path/to/comics

--comics: a folder with comic1/p1.png ... comic500/p4.png (see 20_comic_residual_by_layer.py).
--num_pairs is the number of target images (the paper used 100 held-out comic images), each with
3 sources and 9 pairs of languages. The control is the fixed set of 16 random heads of
language/main_experiments/04, patched at the last prompt token only. The paper's images were drawn
by an earlier builder (the shapes at the top center of each panel, as here, but with other shapes
and colors per image), so expect values near the paper's, not the same values.

Paper: the language moves in 99.2% of cases, with 0% copied words. Random heads: 0%. (The paper also
reports MLPs 33-35 held, the language vector, and zeroing on these images; those tests are in
Appendix F and are not repeated here.)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import color_cases_drawn, color_images, comic_color_image, folder_argument, patched_answer
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import all_heads, load_model
from utils.patching import write_tokens


def main():
    comics = folder_argument("--comics", "a folder with comic1/p1.png ... comic500/p4.png")
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    every_head = all_heads(model)
    random_heads = [every_head[i] for i in np.random.default_rng(1016).choice(len(every_head), 16, replace=False)]
    results = Results("comic_language_heads", args)
    draw = lambda spec: comic_color_image(comics, spec)
    for spec in color_images(args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, case in color_cases_drawn(model, spec, draw):
            text = write_tokens(model, target, source, HEADS["comic_language"])
            language = patched_answer(text, case)
            control = patched_answer(write_tokens(model, target, source, random_heads, every_token=False), case)
            results.add(spec["index"], language_heads=language == "language", shape_copied=language in ("word", "concept"),
                        random_heads=control == "language", outcome=language, text=text, **case)
    results.report("language_heads", paper=99.2, label="language heads: target's shape, source's language")
    results.report("shape_copied", paper=0.0, label="language heads: the source's shape was copied")
    results.report("random_heads", paper=0.0, label="random heads: target's shape, source's language")
    results.save()


if __name__ == "__main__":
    main()
