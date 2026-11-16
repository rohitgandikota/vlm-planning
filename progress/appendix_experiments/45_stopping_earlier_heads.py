"""The heads of the earlier steps at the stop decision (Appendix E.1, "Stopping").

This extends progress/main_experiments/15_stopping.py with other head sets. Both runs end on the
written letter W, so the next token is a comma (the list goes on) or the end of the turn. The
target of the pair goes on, its "source_stop" image stops. At the last answer token we copy, from
one run into the other, in both directions:
    binding heads     their head outputs
    relay heads       the values they read from the prompt (shape token to colon token)
    selection heads   the head outputs of the 64 heads found at the comma token for the next panel
                      (Appendix E.7, utils.heads HEADS["selection_output"])
with the same number of random heads from the same layers as control.

    python progress/appendix_experiments/45_stopping_earlier_heads.py --num_pairs 500

Paper: the binding heads' outputs make a finished list go on in 95.9%, the relay heads' values in
84.5%. No small set of the earlier heads makes a continuing list stop. The selection heads make a
finished list go on in 99.5% (random heads 29.5%) and a continuing list stop in 33.7% (random 0.0%).
These numbers come from an earlier, smaller run (200 held-out pairs of an earlier draw of the list
task, 193 valid), and the binding and relay numbers from the head sets of that run. They were not
rerun on the 500 held-out pairs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, stop_directions
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, run_patches

SETS = {"binding_outputs": ("head_output", "binding"), "relay_values": ("values", "relay"),
        "selection_outputs": ("head_output", "selection_output")}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("stopping_earlier_heads", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        case = {}
        for direction, source, target in stop_directions(model, pair):
            prompt = range(target.token("shape"), target.token("colon") + 1)
            names, patches = [], []
            for name, (component, heads) in SETS.items():
                for label, chosen in ((name, HEADS[heads]), (name + "_random_heads", random_like(HEADS[heads], seed=pair["index"]))):
                    names.append(label)
                    patches.append(make_patch(source, target, component, "last_answer", heads=chosen,
                                              positions=prompt if component == "values" else None))
            answers = run_patches(model, target, patches)["answers"]
            # The patch works when the target now makes the source's choice.
            case.update({f"{direction}_{label}": answer == source.answer for label, answer in zip(names, answers)})
        results.add(pair["index"], **case)
    paper = {"finished_list_goes_on_binding_outputs": 95.9, "finished_list_goes_on_relay_values": 84.5,
             "finished_list_goes_on_selection_outputs": 99.5, "finished_list_goes_on_selection_outputs_random_heads": 29.5,
             "continuing_list_stops_selection_outputs": 33.7, "continuing_list_stops_selection_outputs_random_heads": 0.0}
    for direction in ("finished_list_goes_on", "continuing_list_stops"):
        for name in SETS:
            for label in (name, name + "_random_heads"):
                results.report(f"{direction}_{label}", paper=paper.get(f"{direction}_{label}"), label=f"{direction}: {label}")
    results.save()


if __name__ == "__main__":
    main()
