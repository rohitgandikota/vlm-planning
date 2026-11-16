"""What the stop decision reads at the last answer token (Appendix E.1, "Stopping").

This extends progress/main_experiments/15_stopping.py. Both runs end on the written letter W, with
no comma after it, so the next token is a comma (the list goes on) or the end of the turn (it
stops). The target of the pair has matching panels left; its "source_stop" image holds the shape
only in the panel of W, so that list is finished.
    values   the 64 stop-value heads at the last answer token read the other run's values of the
             prompt (shape token to colon token), in both directions, with random heads as control
    blocks   no source: in each run's own run we block attention, and read whether the choice flips

    python progress/appendix_experiments/43_stopping_values_and_blocks.py --num_pairs 500

Paper: the values make a continuing list stop in 88.2% of cases. A finished list goes on when the
later tokens cannot attend to the shape token (99.8%), and when the relay heads cannot attend to
the text at the last letter (69.2%, random heads 0.0%). A continuing list never stops, whichever
attention we block. Every held-out pair counts (both runs make their own choice in 500 of 500).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, stop_directions
from utils.experiment import Results, parse_args
from utils.heads import HEADS, STOP_VALUES_RANKING, random_like
from utils.model import load_model, n_layers
from utils.patching import Patch, make_patch, run_patches


def blocks(run, seed):
    """The attention blocks in one run's own run. Each is a Patch; the last answer token is `run.last`."""
    every_layer, last, shape = range(n_layers(run.model)), run.last, run.token("shape")
    text = range(shape, last)                                      # the shape token up to the token before the letter
    return {
        "shape_token_blind": Patch().block(None, [shape], run.image_positions, layers=every_layer),
        "colon_blind": Patch().block(None, [run.token("colon")], run.image_positions, layers=every_layer),
        "answer_token_blind": Patch().block(None, [last], run.image_positions, layers=every_layer),
        "gaze_heads_blind_at_answer_token": Patch().block(HEADS["gaze"], [last], run.image_positions),
        "random_heads_blind_at_answer_token": Patch().block(random_like(HEADS["gaze"], seed), [last], run.image_positions),
        "relay_heads_cannot_read_text": Patch().block(HEADS["relay"], [last], text),
        "random_heads_cannot_read_text": Patch().block(random_like(HEADS["relay"], seed), [last], text),
        "later_tokens_cannot_read_shape_token": Patch().block(None, range(shape + 1, last + 1), [shape], layers=every_layer),
        "control_later_tokens_cannot_read_token_before_shape": Patch().block(None, range(shape + 1, last + 1), [shape - 1], layers=every_layer),
    }


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("stopping_values_and_blocks", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["stop_values"], seed=pair["index"], exclude=STOP_VALUES_RANKING)
        case = {}
        for name, source, target in stop_directions(model, pair):
            prompt = range(target.token("shape"), target.token("colon") + 1)     # the same positions in both runs
            values = [make_patch(source, target, "values", "last_answer", heads=heads, positions=prompt)
                      for heads in (HEADS["stop_values"], random_heads)]
            own_blocks = blocks(target, pair["index"])
            answers = run_patches(model, target, values + list(own_blocks.values()))["answers"]
            case.update({f"{name}_values": answers[0] == source.answer, f"{name}_values_random_heads": answers[1] == source.answer})
            # In its own run the target flips when it now makes the other run's choice.
            case.update({f"{name}_own_run_{block}": answer == source.answer for block, answer in zip(own_blocks, answers[2:])})
        results.add(pair["index"], **case)
    results.report("continuing_list_stops_values", paper=88.2, label="stop-value heads: a continuing list stops")
    results.report("continuing_list_stops_values_random_heads", label="random heads: a continuing list stops")
    results.report("finished_list_goes_on_values", label="stop-value heads: a finished list goes on")
    results.report("finished_list_goes_on_own_run_later_tokens_cannot_read_shape_token", paper=99.8,
                   label="finished, later tokens blocked from shape token")
    results.report("finished_list_goes_on_own_run_relay_heads_cannot_read_text", paper=69.2, label="finished, relay heads blocked from text")
    results.report("finished_list_goes_on_own_run_random_heads_cannot_read_text", paper=0.0, label="finished, random heads blocked from text")
    for block in ("shape_token_blind", "answer_token_blind", "relay_heads_cannot_read_text", "later_tokens_cannot_read_shape_token"):
        results.report(f"continuing_list_stops_own_run_{block}", paper=0.0, label=f"continuing, {block.replace('_', ' ')}")
    results.save()


if __name__ == "__main__":
    main()
