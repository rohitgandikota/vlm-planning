"""Size does not change the order of a list on a blank canvas (Appendix G.3.2, "Size does not change the order").

A larger shape could stand out more and be named first. We draw 100 random layouts of six shapes on
a 960 x 960 canvas with the asked shape at four or five places, and give the asked shapes clearly
different sizes, each at least 1.5 times the area of the next smaller one, in a random order that
does not depend on the place (utils.dataset.size_layouts). Every layout is drawn in four versions:
shapes and letters of different sizes, only the shapes, only the letters, and one size for all.
The model writes its list. For every pair of asked shapes that the list names, we ask:
    reading order    does the reading-order rule (row height 160 pixels) put them in this order?
    larger shape     is the larger shape named first? (the versions with shapes of different sizes)
    larger letter    is the larger letter named first? (the versions with letters of different sizes)
No patch is used.

    python generalization/main_experiments/25_canvas_shape_size_and_order.py

--num_pairs is the number of layouts (the paper used 100, 400 images).

Paper: the right set of letters in 98.0% to 100.0% of images; the reading-order rule gives the
order of 88.9% to 90.6% of pairs; the larger shape comes first in 49.6% to 49.9% of pairs and the
larger letter in 49.4%. Against the one-size version, different sizes change the share of pairs
with the larger shape first by only 1.1 to 1.4 points.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import SIZE_VERSIONS, list_question, reading_order, size_image, size_layouts
from utils.experiment import Results, parse_args
from utils.model import load_model
from utils.patching import run_model, write_tokens


def main():
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    results = Results("canvas_shape_size_and_order", args)
    for layout in size_layouts(args.start + args.num_pairs)[args.start:]:
        places, letters, rank = layout["places"], layout["letters"], layout["size_rank"]
        reading = reading_order(layout["centers"], places)
        for version in SIZE_VERSIONS:
            run = run_model(model, size_image(layout, version), list_question(layout["ask"]), record=False)
            written = [item.strip() for item in write_tokens(model, run, max_tokens=30).split(",") if item.strip()]
            named = []                                             # the asked places, in the order the list names them
            for letter in written:
                if letter in letters and letters.index(letter) in places and letters.index(letter) not in named:
                    named.append(letters.index(letter))
            results.add(layout["index"], version=version, **{f"{version}_right_set": sorted(written) == sorted(letters[p] for p in places)})
            for first, second in itertools.combinations(named, 2):      # `first` is named before `second`
                results.add(layout["index"], version=version, **{
                    f"{version}_reading_order": reading.index(first) < reading.index(second),
                    f"{version}_larger_shape_first": rank[first] > rank[second],
                    f"{version}_larger_letter_first": rank[first] > rank[second] if version in ("shapes_and_letters", "letters_only") else None})
    for version in SIZE_VERSIONS:
        results.report(f"{version}_right_set", paper="98.0 to 100.0", label=f"{version}: right set of letters")
        results.report(f"{version}_reading_order", paper="88.9 to 90.6", label=f"{version}: pairs in reading order")
        results.report(f"{version}_larger_shape_first", paper="49.6 to 49.9" if version != "one_size" else None,
                       label=f"{version}: larger shape named first")
    results.report("letters_only_larger_letter_first", paper=49.4, label="letters only: larger letter named first")
    results.save()


if __name__ == "__main__":
    main()
