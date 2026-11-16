"""A new search for language heads on a blank canvas agrees with the paper's (Appendix G.3.2, "A new search agrees").

The search of language/main_experiments/00_find_language_heads.py, run again on the colored-shapes
task with the four shapes at random places on one 768 x 768 white canvas (no panels). On the 12
search images, in every source-target case with two languages, we patch the output of one head at
a time at the last prompt token and score it by how much it raises the first token of the target's
shape in the source's language over the same shape in the target's language. The top 16 heads are
the language heads of the canvas. 29_language_heads_on_a_canvas.py tests them on held-out images.

    python generalization/main_experiments/00_find_language_heads_on_a_canvas.py

--num_pairs is the number of search images (the paper used 12). This is a head search and is slow.

Paper: 12 of the 16 heads found on the canvas are language heads of the paper.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import LANGUAGES, canvas_color_image, color_cases_drawn, color_images, shape_word_token
from utils.experiment import Results, parse_args
from utils.heads import HEADS, rank_heads, score_each_head
from utils.model import load_model


def main():
    args = parse_args(__doc__, split="search", num_pairs=12)
    model = load_model(args.model, args.device)
    results = Results("find_language_heads_on_a_canvas", args)
    total, cases = 0, 0
    for spec in color_images("search", args.start + args.num_pairs)[args.start:]:
        for source, target, case in color_cases_drawn(model, spec, canvas_color_image, target_languages=LANGUAGES):
            moved = shape_word_token(model, case["source_language"], case["target_shape"])     # the language moved
            own = shape_word_token(model, case["target_language"], case["target_shape"])       # nothing changed
            total += score_each_head(model, source, target, "head_output", "last", lambda logits: logits[moved] - logits[own])
            cases += 1
        print(f"image {spec['index']} searched ({cases} cases)", flush=True)
    ranking = rank_heads(total / cases)
    shared = sorted(set(ranking[:16]) & set(HEADS["language"]))
    print(f"language heads found on the canvas: {ranking[:16]}")
    print(f"{len(shared)} of them are among the paper's 16 language heads   paper: 12 of 16")
    results.save(found=ranking[:16], ranking=ranking, shared_with_paper=shared, scores=(total / cases).tolist())


if __name__ == "__main__":
    main()
