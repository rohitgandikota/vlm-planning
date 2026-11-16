"""On photographs, larger objects come first (Appendix G.3.6, "Larger objects come first").

On the strips the model lists from left to right. To see what sets the order on a photograph, we let
the model write its full list for every valid COCO list (see progress/main_experiments/16), and take
every pair of annotated categories that the list names. For each pair we ask which rule puts the
earlier one first (utils.dataset.coco_rules): the larger one (the union of its boxes), the one closer
to the center, the left one, the upper one, or size times closeness to the center. A pair that a rule
cannot order is left out for that rule. No patch is used. We also write the list for the photograph
mirrored left to right, and for the photograph darkened slightly (the brightness times 0.92), and
count the pairs, at least a tenth of the image width apart, whose order reverses.

    python generalization/main_experiments/37_coco_order_of_objects.py --coco /path/to/coco

The folder must hold COCO val2014 (annotations/instances_val2014.json and val2014/). --num_pairs is
the number of images; the paper used all 1,304 and reports the 740 valid lists that were not used to
fix the analysis. The lists are written greedily, up to 40 tokens.

Paper: the larger object comes first in 70.8% of pairs; the first item is the largest annotated
category in 63.2% of lists (40.6% by chance); the more central first 62.9%; the left first 53.1%; the
upper first 49.4%; where size and left to right disagree, the larger first 68.7%; size times
closeness to the center 71.4%. Mirroring reverses 13.4% of pairs, darkening 3.9%.
"""
import itertools
import sys
from pathlib import Path

from PIL import ImageEnhance, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import (COCO_QUESTIONS, coco_argument, coco_category_features, coco_image, coco_list_images, coco_rules,
                           coco_written_list, run_coco_list)
from utils.experiment import Results, parse_args
from utils.model import load_model
from utils.patching import run_model, write_tokens

VERSIONS = {"original": lambda image: image, "mirrored": ImageOps.mirror,
            "darkened": lambda image: ImageEnhance.Brightness(image).enhance(0.92)}


def main():
    coco = coco_argument()
    args = parse_args(__doc__, num_pairs=1304)
    if coco is None:
        return
    model = load_model(args.model, args.device)
    results = Results("coco_order_of_objects", args)
    for entry in coco_list_images(coco)[args.start:args.start + args.num_pairs]:
        if run_coco_list(model, coco, entry)[0] is None:                   # keep the valid lists of the paper
            continue
        image, _ = coco_image(coco, entry)
        features = coco_category_features(entry)
        lists = {name: coco_written_list(write_tokens(model, run_model(model, change(image), COCO_QUESTIONS[entry["group"]], record=False),
                                                      max_tokens=40), features) for name, change in VERSIONS.items()}
        named = lists["original"]
        largest = max(features, key=lambda c: features[c]["area"])
        results.add(entry["id"], group=entry["group"], first_is_largest=named[0] == largest if named else None,
                    chance_first_is_largest=1 / len(features), lists=lists)
        for first, second in itertools.combinations(named, 2):              # `first` is named before `second`
            rules = coco_rules(features[first], features[second])
            row = {name: verdict == 1 for name, verdict in rules.items() if verdict != 0}
            if rules["larger"] != 0 and rules["left"] != 0 and rules["larger"] != rules["left"]:
                row["larger_where_size_and_left_disagree"] = rules["larger"] == 1
            if abs(features[first]["x"] - features[second]["x"]) >= 0.1:
                for version in ("mirrored", "darkened"):
                    if first in lists[version] and second in lists[version]:
                        row[f"{version}_reverses"] = lists[version].index(first) > lists[version].index(second)
            results.add(entry["id"], group=entry["group"], **row)
    for name, paper in (("larger", 70.8), ("first_is_largest", 63.2), ("more_central", 62.9), ("left", 53.1), ("upper", 49.4),
                        ("larger_where_size_and_left_disagree", 68.7), ("size_times_centrality", 71.4),
                        ("mirrored_reverses", 13.4), ("darkened_reverses", 3.9)):
        results.report(name, paper=paper)
    chance = [row["chance_first_is_largest"] for row in results.rows if row.get("first_is_largest") is not None]
    print(f"chance that the first item is the largest: {100 * sum(chance) / max(len(chance), 1):.1f}%   paper: 40.6")
    results.save()


if __name__ == "__main__":
    main()
