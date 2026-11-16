"""On natural images, the comma token finds the written object again by itself (Appendix G.3.6, Table 17).

The COCO list questions of progress/main_experiments/16 ("What animals are in this image?" and so
on). On a valid list the model wrote A, then a comma, and names B next. With no patch, we block:
    A's tokens from A's region (the union of its boxes)       -> B is still named (unchanged)
    A's tokens from the whole image                           -> unchanged
    the comma token from the image                            -> unchanged
    A's tokens and the comma token from the image             -> unchanged
    B's region from every text token, in every layer          -> unchanged
each in every head and layer. We also read the attention of the 16 binding heads at A's last token,
per image token, on A's region and on B's region, against 16 random heads from the same layers.
The comma blocked from A's tokens and the gaze heads at the comma are the main experiments
progress/main_experiments/16 and gaze/main_experiments/07.

    python generalization/main_experiments/36_coco_written_word.py --coco /path/to/coco

The folder must hold COCO val2014 (annotations/instances_val2014.json and val2014/). --num_pairs is
the number of images; the paper uses all 1,304, and 840 of them give a valid list. The random heads
are drawn per image with seed (COCO image id mod 97), as in the paper's run.

Paper (all groups): A blocked from its region 94.2%, from the image 85.7%; comma blocked from the
image 68.6%; both 62.7%; B's region hidden from the first layer 70% unchanged. The binding heads look
at A more than at B in 85.5% of lists (median factor 2.6), random heads in 69.4% (factor 1.4).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import coco_argument, coco_list_images, run_coco_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model, n_layers
from utils.patching import Patch, run_patches


def attention_per_token(run, heads, token, positions):
    """Attention of `heads` at one token, summed over the heads and averaged over these image tokens."""
    return float(sum(run.attention(layer, head, token)[positions].mean() for layer, head in heads))


def main():
    coco = coco_argument()
    args = parse_args(__doc__, num_pairs=1304)
    if coco is None:
        return
    model = load_model(args.model, args.device)
    every_layer = range(n_layers(model))
    results = Results("coco_written_word", args)
    for entry in coco_list_images(coco)[args.start:args.start + args.num_pairs]:
        run, case = run_coco_list(model, coco, entry)
        if run is None:
            results.add(entry["id"], group=entry["group"], valid_list=False, why=case)
            continue
        word, comma, image = run.token("first_item"), [run.token("last_comma")], run.image_positions
        region_a, region_b = case["region_positions"][case["first"]], case["region_positions"][case["second"]]
        patches = [Patch().block(None, word, region_a, layers=every_layer),
                   Patch().block(None, word, image, layers=every_layer),
                   Patch().block(None, comma, image, layers=every_layer),
                   Patch().block(None, word + comma, image, layers=every_layer),
                   Patch().block(None, range(run.first_text, run.last + 1), region_b, layers=every_layer)]
        unchanged = [answer == case["second_token"] for answer in run_patches(model, run, patches)["answers"]]
        last_word_token = word[-1]
        on = lambda heads: [attention_per_token(run, heads, last_word_token, region) for region in (region_a, region_b)]
        binding, control = on(HEADS["binding"]), on(random_like(HEADS["binding"], seed=entry["id"] % 97))
        results.add(entry["id"], group=entry["group"], valid_list=True, word_blind_to_own_region=unchanged[0],
                    word_blind_to_image=unchanged[1], comma_blind_to_image=unchanged[2], word_and_comma_blind=unchanged[3],
                    next_region_hidden=unchanged[4], binding_heads_more_on_A=binding[0] > binding[1],
                    random_heads_more_on_A=control[0] > control[1], binding_ratio=binding[0] / max(binding[1], 1e-12))
    results.report("valid_list", paper=64.4, label="images with a valid list")
    for name, paper in (("word_blind_to_own_region", 94.2), ("word_blind_to_image", 85.7), ("comma_blind_to_image", 68.6),
                        ("word_and_comma_blind", 62.7), ("next_region_hidden", 70.0), ("binding_heads_more_on_A", 85.5),
                        ("random_heads_more_on_A", 69.4)):
        results.report(name, paper=paper)
    ratios = sorted(row["binding_ratio"] for row in results.rows if row.get("valid_list"))
    if ratios:
        print(f"median ratio of the binding heads' attention, A over B: {ratios[len(ratios) // 2]:.2f}   paper: 2.6")
    results.save()


if __name__ == "__main__":
    main()
