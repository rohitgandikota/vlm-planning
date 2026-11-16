"""A single vector sets the language (Appendix F.6, Table 16).

No source run is used. For each language L (German, French, Chinese) we build one vector from the 12
search images: the mean, over the images, of the difference between what the 16 language heads
together write into the residual stream at the last prompt token when the question is asked in L
and when it is asked in English. Then, on the held-out images:
    to L        the vector is added to the residual entering layer 24 at the last prompt token of
                the English run (scaled by 0.5 for German and French, 1.0 for Chinese); counts when
                the answer names the concept in L
    random      a random vector of the same length per dimension, added in the same way (control)
    back        the vector (scale 1.0) is subtracted from the German and French runs; counts when
                the answer turns English
The edit acts on the last prompt token, which every written token then reads.

    python language/appendix_experiments/03_single_vector_sets_language.py --num_pairs 500
    python language/appendix_experiments/03_single_vector_sets_language.py --task objects --num_pairs 100

Paper, colored shapes: French 100.0%, Chinese 100.0%, German 75.0%; subtracted: English 100.0%.
Objects (an earlier run on 100 images): 15.0%, 100.0%, 36.0%; subtracted 100.0%.
The paper also adds the same vectors to text-only prompts with no image (65.0 / 100.0 / 60.0%);
this release runs every prompt with an image, so that part is not included.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import task_argument
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import write_with_patch, written_by_heads

LAYER = 24
SCALE = {"de": 0.5, "fr": 0.5, "zh": 1.0}
PAPER = {"colors": {"de": 75.0, "fr": 100.0, "zh": 100.0, "back": 100.0}, "objects": {"de": 36.0, "fr": 15.0, "zh": 100.0, "back": 100.0}}


def add_to_residual(run, vector):
    """An edit for `write_with_patch`: add `vector` to the residual entering LAYER at the last prompt token."""
    return lambda patch, token: patch.residual(LAYER, token, run.residual(LAYER, token) + vector)


def main():
    task = task_argument()
    args = parse_args(__doc__, num_pairs=500 if task.name == "colors" else 100)
    model = load_model(args.model, args.device)
    heads = HEADS["language" if task.name == "colors" else "language_objects"]
    written = lambda spec, language: written_by_heads(model, task.run(model, spec, language), heads, "last")
    search = task.images("search")
    vectors = {language: sum(written(spec, language) - written(spec, "en") for spec in search) / len(search) for language in SCALE}
    results = Results(f"single_vector_sets_language_{task.name}", args)
    for spec in task.images(args.split, args.start + args.num_pairs)[args.start:]:
        concept, english = task.concept(spec), task.run(model, spec, "en")
        generator = torch.Generator().manual_seed(spec["index"])
        row = {}
        for language, scale in SCALE.items():
            vector = vectors[language]
            noise = torch.randn(vector.numel(), generator=generator).to(vector) * vector.norm() / vector.numel() ** 0.5
            for name, added in ((language, scale * vector), (f"random_{language}", scale * noise)):
                text = write_with_patch(model, english, add_to_residual(english, added))
                row[name] = task.answer_language(text, concept) == language
        results.add(spec["index"], **row)
        for language in ("de", "fr"):
            run = task.run(model, spec, language)
            text = write_with_patch(model, run, add_to_residual(run, -vectors[language]))
            results.add(spec["index"], source_language=language, back=task.answer_language(text, concept) == "en")
    for name in ("fr", "zh", "de", "back"):
        results.report(name, paper=PAPER[task.name][name])
    for language in SCALE:
        results.report(f"random_{language}", label=f"random vector: {language}")
    results.save()


if __name__ == "__main__":
    main()
