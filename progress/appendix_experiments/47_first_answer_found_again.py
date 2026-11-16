"""On the list task the colon token binds the question to the image by itself (Appendix E.3, "The first answer").

No source is patched in. Nothing is written, so the colon token gives the first answer, the letter
at r1. In the target's own run we block attention in one place at a time and read whether the first
answer survives:
    relay heads cannot read the text   the 16 relay heads at the colon token, from the shape token
                                       up to the token before the colon (random heads as control)
    shape token blind                  the shape token cannot attend to the image, in every layer
    colon token blind                  the colon token cannot attend to the image, in every layer
On the single-answer task the same relay block leaves the answer in 7.8% of cases (Appendix C).

    python progress/appendix_experiments/47_first_answer_found_again.py --num_pairs 500

Paper: the first answer survives in 66.8% of cases with the relay heads blocked, in 67.2% with the
shape token blind, and in 99.4% with the colon token blind. A pair counts when the target gives its
r1 letter and its "source_first" image its own first letter with no patch (500 of 500).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model, n_layers
from utils.patching import Patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    every_layer = range(n_layers(model))
    results = Results("first_answer_found_again", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        own = " " + matching_letters(pair["target"])[0]
        target = run_list(model, pair["target"], [], record=False)
        source = run_list(model, pair["source_first"], [], record=False)
        if target.answer != own or source.answer != " " + matching_letters(pair["source_first"])[0]:
            continue
        colon, shape, image = target.token("colon"), target.token("shape"), target.image_positions
        text = range(shape, colon)                                 # the shape token up to the token before the colon
        patches = {
            "relay_heads_cannot_read_text": Patch().block(HEADS["relay"], [colon], text),
            "random_heads_cannot_read_text": Patch().block(random_like(HEADS["relay"], pair["index"]), [colon], text),
            "shape_token_blind": Patch().block(None, [shape], image, layers=every_layer),
            "colon_token_blind": Patch().block(None, [colon], image, layers=every_layer),
            "later_tokens_cannot_read_shape_token": Patch().block(None, range(shape + 1, colon + 1), [shape], layers=every_layer),
        }
        answers = run_patches(model, target, list(patches.values()))["answers"]
        results.add(pair["index"], **{name: answer == own for name, answer in zip(patches, answers)})
    results.report("relay_heads_cannot_read_text", paper=66.8, label="relay heads blocked from the text: survives")
    results.report("random_heads_cannot_read_text", label="random heads blocked from the text: survives")
    results.report("shape_token_blind", paper=67.2, label="shape token blind: survives")
    results.report("colon_token_blind", paper=99.4, label="colon token blind: survives")
    results.report("later_tokens_cannot_read_shape_token", label="later tokens blocked from the shape token: survives")
    results.save()


if __name__ == "__main__":
    main()
