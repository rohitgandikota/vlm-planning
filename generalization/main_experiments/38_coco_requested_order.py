"""A requested order changes the list on a photograph (Appendix G.3.6, "A requested order on photographs").

We add one sentence to the COCO list question (utils.dataset.COCO_ORDER_SENTENCES) and compare the
new list with the list from the original question. Each pair of annotated categories named in both
lists keeps its order or reverses. We also measure, for each sentence, the share of named pairs with
the larger object first and with the left object first (as in 37_coco_order_of_objects.py).
    neutral          " List them in any order."
    most_salient     " List them starting with the most salient one."
    most_prominent   " Start with the most prominent one."
    left_to_right    " List them from left to right."
No patch is used.

    python generalization/main_experiments/38_coco_requested_order.py --coco /path/to/coco

The folder must hold COCO val2014 (annotations/instances_val2014.json and val2014/). --num_pairs is
the number of valid lists used (the paper used 100). Lists are written greedily, up to 40 tokens.

Paper: the neutral sentence reverses 4.3% of pairs, the most salient 15.2%, the most prominent 16.1%,
left to right 21.0%. With the most prominent one first, the share of pairs with the larger object
first rises by 9.9 points over the neutral sentence; with left to right, the share in left-to-right
order rises by 14.7 points.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import (COCO_ORDER_SENTENCES, COCO_QUESTIONS, coco_argument, coco_category_features, coco_image,
                           coco_list_images, coco_rules, coco_written_list, run_coco_list)
from utils.experiment import Results, parse_args
from utils.model import load_model
from utils.patching import run_model, write_tokens


def main():
    coco = coco_argument()
    args = parse_args(__doc__, num_pairs=100)
    if coco is None:
        return
    model = load_model(args.model, args.device)
    results = Results("coco_requested_order", args)
    used = 0
    for entry in coco_list_images(coco)[args.start:]:
        if used == args.num_pairs:
            break
        if run_coco_list(model, coco, entry)[0] is None:
            continue
        used += 1
        image, _ = coco_image(coco, entry)
        features = coco_category_features(entry)
        write = lambda sentence: coco_written_list(write_tokens(model, run_model(
            model, image, COCO_QUESTIONS[entry["group"]] + sentence, record=False), max_tokens=40), features)
        lists = {name: write(sentence) for name, sentence in {"original": "", **COCO_ORDER_SENTENCES}.items()}
        for name, named in lists.items():
            for first, second in itertools.combinations(named, 2):           # `first` is named before `second`
                rules = coco_rules(features[first], features[second])
                row = {f"{name}_larger_first": rules["larger"] == 1 if rules["larger"] else None,
                       f"{name}_left_first": rules["left"] == 1 if rules["left"] else None}
                if name != "original" and first in lists["original"] and second in lists["original"]:
                    row[f"{name}_reverses"] = lists["original"].index(first) > lists["original"].index(second)
                results.add(entry["id"], group=entry["group"], **row)
    for name, paper in (("neutral", 4.3), ("most_salient", 15.2), ("most_prominent", 16.1), ("left_to_right", 21.0)):
        results.report(f"{name}_reverses", paper=paper, label=f"{name}: pairs that reverse")
    for name in ("original", *COCO_ORDER_SENTENCES):
        results.report(f"{name}_larger_first", label=f"{name}: larger object first")
        results.report(f"{name}_left_first", label=f"{name}: left object first")
    share = lambda condition: results.rate(condition)["percent"] or 0.0
    print(f"most prominent minus neutral, larger first: {share('most_prominent_larger_first') - share('neutral_larger_first'):+.1f} points   paper: +9.9")
    print(f"left to right minus neutral, left first: {share('left_to_right_left_first') - share('neutral_left_first'):+.1f} points   paper: +14.7")
    results.save()


if __name__ == "__main__":
    main()
