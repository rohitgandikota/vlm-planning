"""Blocking attention in the target's own run, at the second answer (Appendix D.4, Table "many knockouts").

The target has written W, its letter at r1, so its next answer is its letter at r2. No source is
used. We block one set of attention links, read the next letter, and count two outcomes: the own
answer is kept, or the model writes W again. "Blocked from the image" means from every image token,
in every head of every layer, unless heads are named. Each row of the table is one patch below.
Two rows are also main experiments (progress/main_experiments/06_comma_blocked_from_answer_token.py).

    python progress/appendix_experiments/09_blocking_at_the_second_answer.py --num_pairs 500

Paper (own answer kept / W again):
    nothing blocked                                                         100.0 /  0.0
    shape token blocked from the image                                       68.8 /  0.0
    later tokens cannot attend to the shape token                            66.4 /  0.0
      control: later tokens cannot attend to the token before it            100.0 /  0.0
    relay heads cannot attend to the text at the comma token                 65.6 /  0.0
      random heads cannot attend to the text at the comma token             100.0 /  0.0
    answer token blocked from the image                                      91.4 /  5.2
    comma token blocked from the image                                       98.6 /  0.0
    answer token and comma token blocked from the image                      18.2 /  0.0
    comma token cannot attend to the answer token                            27.4 / 67.8
      control: comma token cannot attend to the colon token                  99.8 /  0.0
    only the shape token, the answer token, and the gaze heads at the
      comma token see the image                                              86.8 /  0.0
    only the shape token and the gaze heads at the comma token see the image 43.8 /  0.0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import all_heads, load_model, n_layers
from utils.patching import Patch, run_patches

PAPER = {"nothing": (100.0, 0.0), "shape_from_image": (68.8, 0.0), "later_tokens_from_shape": (66.4, 0.0),
         "later_tokens_from_token_before_shape": (100.0, 0.0), "relay_heads_from_text": (65.6, 0.0),
         "random_heads_from_text": (100.0, 0.0), "answer_from_image": (91.4, 5.2), "comma_from_image": (98.6, 0.0),
         "answer_and_comma_from_image": (18.2, 0.0), "comma_from_answer": (27.4, 67.8), "comma_from_colon": (99.8, 0.0),
         "only_shape_answer_and_gaze_heads_see_image": (86.8, 0.0), "only_shape_and_gaze_heads_see_image": (43.8, 0.0)}


def blocks(model, target, seed):
    """Every patch of the table, by name."""
    every = range(n_layers(model))
    shape, colon, answer, comma = (target.token(name) for name in ("shape", "colon", "last_answer", "last_comma"))
    image, later = target.image_positions, list(range(shape + 1, comma + 1))      # later: every token after the shape token
    text = list(range(shape, comma))                                               # the shape token up to the answer token
    gaze = set(HEADS["gaze"])
    not_gaze = [head for head in all_heads(model) if head not in gaze]
    only = lambda keep: (Patch().block(None, [t for t in later if t not in keep], image, layers=every)
                         .block(not_gaze, [comma], image))
    return {
        "nothing": Patch(),
        "shape_from_image": Patch().block(None, [shape], image, layers=every),
        "later_tokens_from_shape": Patch().block(None, later, [shape], layers=every),
        "later_tokens_from_token_before_shape": Patch().block(None, later, [shape - 1], layers=every),
        "relay_heads_from_text": Patch().block(HEADS["relay"], [comma], text),
        "random_heads_from_text": Patch().block(random_like(HEADS["relay"], seed=seed), [comma], text),
        "answer_from_image": Patch().block(None, [answer], image, layers=every),
        "comma_from_image": Patch().block(None, [comma], image, layers=every),
        "answer_and_comma_from_image": Patch().block(None, [answer, comma], image, layers=every),
        "comma_from_answer": Patch().block(None, [comma], [answer], layers=every),
        "comma_from_colon": Patch().block(None, [comma], [colon], layers=every),
        "only_shape_answer_and_gaze_heads_see_image": only(keep=(answer, comma)),
        "only_shape_and_gaze_heads_see_image": only(keep=(comma,)),
    }


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("blocking_at_the_second_answer", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        own_next, W = matching_letters(pair["target"])[1], pair["written"]
        target = run_list(model, pair["target"], [W])
        patches = blocks(model, target, seed=pair["index"])
        answers = run_patches(model, target, list(patches.values()))["answers"]
        row = {}
        for name, answer in zip(patches, answers):
            row[f"{name}_own"], row[f"{name}_again"] = answer == " " + own_next, answer == " " + W
        results.add(pair["index"], **row)
    for name, (own, again) in PAPER.items():
        results.report(f"{name}_own", paper=own, label=f"{name}: own answer kept")
        results.report(f"{name}_again", paper=again, label=f"{name}: W again")
    results.save()


if __name__ == "__main__":
    main()
