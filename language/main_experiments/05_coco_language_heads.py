"""On COCO photographs, the language heads carry the language of the answer (Section 6.2).

We ask a natural list question about a photograph in German, for example "Welche Tiere sind auf
diesem Bild?", with nothing prefilled, and read the first token of the answer. We keep the images
where this token starts a German name of an annotated object. Then we set the output of each of
the 16 language heads, at the last prompt token, to its average over 12 other photographs asked
the same kind of question in English. No source run is used. If these heads carry the language,
the first token should now start an English name.

    python language/main_experiments/05_coco_language_heads.py --coco /path/to/coco

The folder must hold COCO val2014 (annotations/instances_val2014.json and val2014/). --num_pairs
is the number of images; the paper uses the 904 animal, vehicle, and furniture images, and the
answer starts in German on 398 of them. The control sets 16 random heads of the same layers to
their English averages.

Paper: 96.5% of the German answers turn English. With random heads, 97.0% stay German. The paper
ran this with the language heads of an earlier search, which share 13 of their 16 heads with the
final list. This file uses that earlier list (HEADS["language_coco"]).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import COCO_QUESTIONS, COCO_QUESTIONS_GERMAN, coco_argument, coco_image, coco_language, coco_list_images
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import Patch, run_model, run_patches


def english_averages(model, coco, heads):
    """The average output of each head at the last prompt token, over the 12 photographs set aside, asked in English."""
    runs = [run_model(model, coco_image(coco, entry)[0], COCO_QUESTIONS[entry["group"]], prefill="")
            for entry in coco_list_images(coco, part="set_aside")]
    return {(layer, head): sum(run.head_output(layer, head, "last") for run in runs) / len(runs) for layer, head in heads}


def main():
    coco = coco_argument()
    args = parse_args(__doc__, num_pairs=904)
    if coco is None:
        return
    model = load_model(args.model, args.device)
    language_heads, random_heads = HEADS["language_coco"], random_like(HEADS["language_coco"], seed=7)
    average = english_averages(model, coco, language_heads + random_heads)
    results = Results("coco_language_heads", args)
    images = [entry for entry in coco_list_images(coco) if entry["group"] in COCO_QUESTIONS_GERMAN]
    for entry in images[args.start:args.start + args.num_pairs]:
        image, regions = coco_image(coco, entry)
        run = run_model(model, image, COCO_QUESTIONS_GERMAN[entry["group"]], prefill="", record=False)
        if coco_language(run.answer, regions) != "german":           # the answer does not start in German
            results.add(entry["id"], group=entry["group"], german_answer=False, answer=run.answer)
            continue
        patches = []
        for heads in (language_heads, random_heads):
            patch = Patch()
            for layer, head in heads:
                patch.head_output(layer, head, run.last, average[(layer, head)])
            patches.append(patch)
        with_language_heads, with_random_heads = run_patches(model, run, patches)["answers"]
        results.add(entry["id"], group=entry["group"], german_answer=True, answer=run.answer,
                    language_heads=coco_language(with_language_heads, regions) == "english",
                    random_heads=coco_language(with_random_heads, regions) == "german",
                    answer_language_heads=with_language_heads, answer_random_heads=with_random_heads)
    results.report("german_answer", label="answers that start in German")
    results.report("language_heads", paper=96.5, label="language heads set to English: answer turns English")
    results.report("random_heads", paper=97.0, label="random heads set to English: answer stays German")
    results.save()


if __name__ == "__main__":
    main()
