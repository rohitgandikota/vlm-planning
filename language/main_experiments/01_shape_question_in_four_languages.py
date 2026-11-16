"""The model answers the shape question in the language of the question (Section 6.2).

The colored-shapes task: four panels, one filled shape in each, no letters, exactly one shape
cyan. We ask for the shape of the cyan marker in English, German, French, and Chinese, with
nothing prefilled, and let the model write its answer. No patch. An answer counts when the word
for the right shape appears in the language of the question.

    python language/main_experiments/01_shape_question_in_four_languages.py --num_pairs 500

--num_pairs is the number of images (the paper uses 500, asked in four languages: 2,000 answers).

Paper: 100.0% of answers name the right shape in the question's language.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.patching import write_tokens
from utils.dataset import LANGUAGES, answer_language, color_images, run_color_question
from utils.experiment import Results, parse_args
from utils.model import load_model


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("shape_question_in_four_languages", args)
    for spec in color_images(args.split, args.start + args.num_pairs)[args.start:]:
        for language in LANGUAGES:
            text = write_tokens(model, run_color_question(model, spec, language, record=False))
            results.add(spec["index"], right_shape_and_language=answer_language(text, spec["shape"]) == language,
                        language=language, shape=spec["shape"], text=text)
    results.report("right_shape_and_language", paper=100.0, label="right shape in the question's language")
    results.save()


if __name__ == "__main__":
    main()
