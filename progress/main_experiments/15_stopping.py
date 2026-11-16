"""The same step decides when to stop (Section 5.4).

At the last written letter the model chooses between a comma (the list goes on) and the end of its
turn (the list stops). We take two runs that end on the same written letter W. The target of the
pair has three more matching panels and goes on. The other image holds the shape only in the
panel of W, so its list is finished and it stops. We copy the outputs of the 64 stop heads at the
last answer token from one run into the other, in both directions.

    python progress/main_experiments/15_stopping.py --num_pairs 500

Paper: the patch makes a finished list go on in 100.0% of cases and a continuing list stop in
96.8%. The same number of random heads from the same layers give 0.2% and 0.0%. The random heads
are drawn outside the best 128 heads of the search (progress/main_experiments/00_find_stop_heads.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, stop_directions
from utils.experiment import Results, parse_args
from utils.heads import HEADS, STOP_HEAD_RANKING, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("stopping", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["stop"], seed=pair["index"], exclude=STOP_HEAD_RANKING)
        case = {}
        for name, source, target in stop_directions(model, pair):
            patches = [make_patch(source, target, "head_output", "last_answer", heads=heads)
                       for heads in (HEADS["stop"], random_heads)]
            patched, control = run_patches(model, target, patches)["answers"]
            # The patch works when the target now makes the source's choice (a comma, or the end of the turn).
            case.update({name: patched == source.answer, name + "_random_heads": control == source.answer})
        results.add(pair["index"], **case)
    results.report("finished_list_goes_on", paper=100.0, label="stop heads: a finished list goes on")
    results.report("continuing_list_stops", paper=96.8, label="stop heads: a continuing list stops")
    results.report("finished_list_goes_on_random_heads", paper=0.2, label="random heads: a finished list goes on")
    results.report("continuing_list_stops_random_heads", paper=0.0, label="random heads: a continuing list stops")
    results.save()


if __name__ == "__main__":
    main()
