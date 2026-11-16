"""On COCO photographs, the comma token reads progress from the written word (Section 5.4).

We ask a natural list question about a photograph, for example "What animals are in this image?".
The model writes its list. We keep the lists whose first two items are two different annotated
COCO categories, A then B, and prefill "Answer: A,". The model then names B. We block the
attention from the comma token to every token of the written word A, in every layer and every
head. If the comma token learns from the written word what is done, the model should now name A
again. As a control we block the comma token from the colon token instead.

    python progress/main_experiments/16_coco_object_blocked.py --coco /path/to/coco

The folder must hold COCO val2014 in the standard layout: annotations/instances_val2014.json and
the images in val2014/. Nothing is downloaded. --num_pairs is the number of images; the paper uses
all 1,304 and 840 of them give a valid list.

Paper: the model names the written object again in 79.6% of valid lists. With the colon token
blocked instead, 98.0% of answers are unchanged. Answers are compared by the first token of a name.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import coco_argument, coco_list_images, run_coco_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import Patch, run_patches


def main():
    coco = coco_argument()
    args = parse_args(__doc__, num_pairs=1304)
    if coco is None:
        return
    model = load_model(args.model, args.device)
    results = Results("coco_object_blocked", args)
    for entry in coco_list_images(coco)[args.start:args.start + args.num_pairs]:
        run, case = run_coco_list(model, coco, entry)
        if run is None:                                              # no valid list for this image
            results.add(entry["id"], valid_list=False, why=case)
            continue
        comma, every_layer = [run.token("last_comma")], range(n_layers(model))
        patches = [Patch().block(None, comma, run.token("first_item"), layers=every_layer),
                   Patch().block(None, comma, [run.token("colon")], layers=every_layer)]
        blocked, control = run_patches(model, run, patches)["answers"]
        results.add(entry["id"], valid_list=True, group=entry["group"], answer=blocked,
                    names_it_again=blocked == case["first_token"] and blocked != case["second_token"],
                    colon_blocked_unchanged=control == case["second_token"])
    results.report("valid_list", label="images with a valid list")
    results.report("names_it_again", paper=79.6, label="written word blocked: names it again")
    results.report("colon_blocked_unchanged", paper=98.0, label="colon token blocked: answer unchanged")
    results.save()


if __name__ == "__main__":
    main()
