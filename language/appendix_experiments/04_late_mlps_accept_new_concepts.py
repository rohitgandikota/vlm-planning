"""The late MLPs accept concepts they have not been shown (Appendix F.6).

No source run is used. Into a run asked in German, French, or Chinese we add the vocabulary
direction of a new English noun at the last prompt token: the output-embedding row of the noun's
first token minus the row of the shown concept's English name (both made unit length), scaled to
the length of the residual vector it is added to. We add it to the residual entering layer 26, or
late, entering layer 32, after the language heads have acted. The edit acts on the last prompt
token, which every written token then reads. A case counts as "translated" when the answer names
the noun in the question's language, and as "English" when it names the English noun.

    python language/appendix_experiments/04_late_mlps_accept_new_concepts.py --num_pairs 500
    python language/appendix_experiments/04_late_mlps_accept_new_concepts.py --task objects --num_pairs 100

Each image gives 3 languages x 10 nouns (7 on the object task) x 2 layers.

Paper: added at layer 26, the noun in the question's language in 74.6% of cases (objects 67.0%,
from an earlier run on 100 images), the English word in 4.6%. Added at layer 32, the English word in
69.2%, the translated word in 3.8%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import new_nouns, noun_language, task_argument
from utils.experiment import Results, parse_args
from utils.model import load_model
from utils.patching import write_with_patch

LAYERS = {"layer_26": 26, "layer_32": 32}


def word_row(model, word):
    """The output-embedding row of the first token of " word", made unit length."""
    row = model.lm_head.weight[model.processor.tokenizer.encode(" " + word, add_special_tokens=False)[0]].float()
    return row / row.norm()


def main():
    task = task_argument()
    args = parse_args(__doc__, num_pairs=500 if task.name == "colors" else 100)
    model = load_model(args.model, args.device)
    results = Results(f"late_mlps_accept_new_concepts_{task.name}", args)
    for spec in task.images(args.split, args.start + args.num_pairs)[args.start:]:
        shown = word_row(model, task.concept(spec))
        for language in ("de", "fr", "zh"):
            run = task.run(model, spec, language)
            for noun in new_nouns(task):
                direction = word_row(model, noun) - shown
                row = {}
                for name, layer in LAYERS.items():
                    base = run.residual(layer, run.last)
                    added = base + base.norm() * direction / direction.norm()
                    text = write_with_patch(model, run, lambda patch, token: patch.residual(layer, token, added))
                    row[f"{name}_translated"] = noun_language(text, noun) == language
                    row[f"{name}_english"] = noun_language(text, noun) == "en"
                results.add(spec["index"], language=language, noun=noun, **row)
    paper = {"colors": (74.6, 4.6, 69.2, 3.8), "objects": (67.0, None, None, None)}[task.name]
    results.report("layer_26_translated", paper=paper[0], label="added at layer 26: noun in the question's language")
    results.report("layer_26_english", paper=paper[1], label="added at layer 26: English noun")
    results.report("layer_32_english", paper=paper[2], label="added at layer 32: English noun")
    results.report("layer_32_translated", paper=paper[3], label="added at layer 32: noun in the question's language")
    results.save()


if __name__ == "__main__":
    main()
