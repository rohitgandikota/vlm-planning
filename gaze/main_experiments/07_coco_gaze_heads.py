"""On COCO photographs, the gaze heads look at the object named next (Section 6.2).

We ask a natural list question about a photograph, for example "What animals are in this image?".
The model writes its list. We keep the lists whose first two items are two different annotated
COCO categories, prefill the first item and a comma, and look at the comma token, where the next
item is chosen. No patch: we measure the attention of the gaze heads there, per image token, on
the boxes of the object named next and on the boxes of the object just named.

    python gaze/main_experiments/07_coco_gaze_heads.py --coco /path/to/coco

The folder must hold COCO val2014 (annotations/instances_val2014.json and val2014/). --num_pairs
is the number of images; the paper uses all 1,304 and 840 of them give a valid list.

Paper: the gaze heads attend more to the object named next than to the one just named in 83.6% of
lists (median ratio 2.6). The paper ran this with the gaze heads of an earlier search, which share
31 of their 32 heads with the final list. This file uses that earlier list (HEADS["gaze_coco"]).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import coco_argument, coco_list_images, run_coco_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model


def attention_per_token(run, heads, token, positions):
    """Attention of `heads` at one token, summed over the heads and averaged over these image tokens."""
    return float(sum(run.attention(layer, head, token)[positions].mean() for layer, head in heads))


def main():
    coco = coco_argument()
    args = parse_args(__doc__, num_pairs=1304)
    if coco is None:
        return
    model = load_model(args.model, args.device)
    results = Results("coco_gaze_heads", args)
    for entry in coco_list_images(coco)[args.start:args.start + args.num_pairs]:
        run, case = run_coco_list(model, coco, entry)
        if run is None:                                              # no valid list for this image
            results.add(entry["id"], group=entry["group"], valid_list=False, why=case)
            continue
        named, following = (attention_per_token(run, HEADS["gaze_coco"], "last_comma", case["region_positions"][c])
                            for c in (case["first"], case["second"]))
        results.add(entry["id"], group=entry["group"], valid_list=True, more_on_next_object=following > named,
                    on_named=named, on_next=following)
    results.report("valid_list", label="images with a valid list")
    results.report("more_on_next_object", paper=83.6, label="gaze heads: more attention on the object named next")
    ratios = sorted(row["on_next"] / row["on_named"] for row in results.rows if row.get("on_named"))
    if ratios:
        print(f"median ratio of attention, next object over named object: {ratios[len(ratios) // 2]:.2f}   paper: 2.57")
    results.save()


if __name__ == "__main__":
    main()
