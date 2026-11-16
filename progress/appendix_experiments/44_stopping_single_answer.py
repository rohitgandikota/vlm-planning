"""On the single-answer task the instruction decides the stop (Appendix E.1, "Stopping").

We take the one-answer task and write the model's own answer letter after "Answer:", so the next
token is its choice after the letter. With no patch it ends its turn. We then block attention in
the run's own run, in eleven ways, and read the next token again. If the stop were decided from the
image as in a list, some block would make the model go on with a comma.

    python progress/appendix_experiments/44_stopping_single_answer.py --num_pairs 500

Paper: the model ends its turn in 100.0% of runs, whichever attention we block. Both images of
every held-out one-answer pair are used (1,000 runs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import one_answer_image, one_answer_pairs, one_answer_question
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model, n_layers
from utils.patching import Patch, run_model, run_patches


def blocks(run, shape, seed):
    """The blocks, as Patches. The run ends "...Answer: X": the colon token, then the answer token."""
    every_layer, image = range(n_layers(run.model)), run.image_positions
    letter, colon = run.last, run.last - 1
    binding = HEADS["binding_query_key"]
    return {
        "shape_token_to_colon_blind": Patch().block(None, range(shape, colon), image, layers=every_layer),
        "shape_token_blind": Patch().block(None, [shape], image, layers=every_layer),
        "control_earlier_text_blind": Patch().block(None, range(run.first_text, shape), image, layers=every_layer),
        "answer_token_blind": Patch().block(None, [letter], image, layers=every_layer),
        "binding_heads_blind_at_answer_token": Patch().block(binding, [letter], image),
        "random_heads_blind_at_answer_token": Patch().block(random_like(binding, seed), [letter], image),
        "relay_heads_cannot_read_text": Patch().block(HEADS["relay"], [letter], range(shape, letter)),
        "random_heads_cannot_read_text": Patch().block(random_like(HEADS["relay"], seed), [letter], range(shape, letter)),
        "answer_token_cannot_read_shape_token": Patch().block(None, [letter], [shape], layers=every_layer),
        "answer_token_cannot_read_colon": Patch().block(None, [letter], [colon], layers=every_layer),
        "answer_and_colon_blind": Patch().block(None, [colon, letter], image, layers=every_layer),
    }


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("stopping_single_answer", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for side_name in ("source", "target"):
            side = pair[side_name]
            letter = side["letters"][side["shapes"].index(side["ask"])]       # the model's own answer
            run = run_model(model, one_answer_image(side), one_answer_question(side["ask"]), prefill="Answer: " + letter, record=False)
            cuts = blocks(run, run.find(" " + side["ask"]), pair["index"])
            answers = run_patches(model, run, [Patch()] + list(cuts.values()))["answers"]
            results.add(pair["index"], side=side_name, no_block_ends=answers[0] == "<end>",
                        **{f"{name}_ends": answer == "<end>" for name, answer in zip(cuts, answers[1:])})
    results.report("no_block_ends", paper=100.0, label="no block: ends its turn")
    for name in ("shape_token_to_colon_blind", "shape_token_blind", "control_earlier_text_blind", "answer_token_blind",
                 "binding_heads_blind_at_answer_token", "random_heads_blind_at_answer_token", "relay_heads_cannot_read_text",
                 "random_heads_cannot_read_text", "answer_token_cannot_read_shape_token", "answer_token_cannot_read_colon",
                 "answer_and_colon_blind"):
        results.report(f"{name}_ends", paper=100.0, label=f"{name.replace('_', ' ')}: ends its turn")
    results.save()


if __name__ == "__main__":
    main()
