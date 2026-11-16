"""Language and gaze heads on a blank canvas (Appendix G.3.2, "A new search agrees").

The colored-shapes task with the four shapes at random places on one white canvas, no panels. As in
language/main_experiments/04 and 03, a target asks in one language about one image and a source in
another language about another image, and while the target writes we copy head outputs from the
source, at the last prompt token and at every written token:
    the 16 language heads found on the canvas    -> the target's shape in the source's language
    the paper's 32 gaze heads                    -> the source's shape in the target's language
The language heads are read from the result file of 00_find_language_heads_on_a_canvas.py if it is
in --output; otherwise the paper's 16 language heads are used, and the file says so.

    python generalization/main_experiments/29_language_heads_on_a_canvas.py

--num_pairs is the number of held-out target images (the paper used 20), each with 3 sources and 9
pairs of languages. The control is the fixed set of 16 random heads of language/main_experiments/04.

Paper: the heads found on the canvas move the language in 100.0% of cases, with 0.0% copied words
(random heads 0.0%). The paper's 32 gaze heads move the concept in 71.5% of cases and the language
in 0.0%. (The late MLPs held, 8.3%, and the heads set to the average of another language, 77.8% and
100.0%, use the tests of Appendix F and are not repeated here.)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import canvas_color_image, color_cases_drawn, color_images, patched_answer
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import all_heads, load_model
from utils.patching import write_tokens


def main():
    args = parse_args(__doc__, num_pairs=20)
    model = load_model(args.model, args.device)
    found = Path(args.output) / "find_language_heads_on_a_canvas.json"
    language_heads = [tuple(h) for h in json.loads(found.read_text())["found"]] if found.exists() else HEADS["language"]
    print("language heads:", "found on the canvas" if found.exists() else "the paper's (no search result found)", flush=True)
    every_head = all_heads(model)
    random_heads = [every_head[i] for i in np.random.default_rng(1016).choice(len(every_head), 16, replace=False)]
    results = Results("language_heads_on_a_canvas", args)
    for spec in color_images(args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, case in color_cases_drawn(model, spec, canvas_color_image):
            language = patched_answer(write_tokens(model, target, source, language_heads), case)
            gaze = patched_answer(write_tokens(model, target, source, HEADS["gaze"]), case)
            control = patched_answer(write_tokens(model, target, source, random_heads, every_token=False), case)
            results.add(spec["index"], language_heads=language == "language", shape_copied=language in ("word", "concept"),
                        random_heads=control == "language", gaze_heads_concept=gaze == "concept",
                        gaze_heads_language=gaze == "language", **case)
    results.report("language_heads", paper=100.0, label="language heads: target's shape, source's language")
    results.report("shape_copied", paper=0.0, label="language heads: the source's shape was copied")
    results.report("random_heads", paper=0.0, label="random heads: target's shape, source's language")
    results.report("gaze_heads_concept", paper=71.5, label="gaze heads: source's shape, target's language")
    results.report("gaze_heads_language", paper=0.0, label="gaze heads: the language changed")
    results.save(language_heads=language_heads)


if __name__ == "__main__":
    main()
