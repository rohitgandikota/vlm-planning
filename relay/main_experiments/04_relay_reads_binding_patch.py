"""Relay heads carry the pointer that the binding heads resolved (Section 4.2).

Two runs of the target are used.
    1. The binding patch: the target with the outputs of the 16 binding heads patched from the
       source at the shape token (03_binding_heads.py). We record this patched run.
    2. An unpatched target, in which the 16 relay heads at the colon token read their values, from
       the shape token and the tokens after it, out of run 1. Nothing else of run 1 is used.
If the relay heads carry the pointer made by the binding heads, run 2 answers K, the target's own
letter at the panel the source points to.

The control in the other direction: keep the binding patch, but let the relay heads read the
target's original values. The target's own answer (outcome T) should come back.

    python relay/main_experiments/04_relay_reads_binding_patch.py --num_pairs 500

Paper: 89.0% of cases give K (random heads from the same layers: 0.0%). With the target's values
kept, 99.8% of cases give T.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, record_patch, run_patches


def binding_patch(source, target):
    """The outputs of the binding heads at the shape token come from the source."""
    return make_patch(source, target, "head_output", "shape", heads=HEADS["binding"])


def relay_values(run, target, heads):
    """At the colon token of the target, the heads read the values of `run`, from the shape token
    through the token before the colon token."""
    read_from = range(target.token("shape"), target.token("colon"))
    return make_patch(run, target, "values", "colon", heads=heads, positions=read_from)


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("relay_reads_binding_patch", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["relay"], seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            patched = record_patch(model, target, binding_patch(source, target))            # run 1
            patches = [relay_values(patched, target, HEADS["relay"]),                       # run 2
                       relay_values(patched, target, random_heads),                         # run 2 with random heads
                       binding_patch(source, target).add(relay_values(target, target, HEADS["relay"]))]   # values kept
            relay, control, kept = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], relay_heads=relay == " " + letters["K"],
                        random_heads=control == " " + letters["K"],
                        target_values_kept=kept == " " + letters["T"], answer=relay)
    results.report("relay_heads", paper=89.0, label="relay heads read the binding patch (K)")
    results.report("random_heads", paper=0.0, label="random heads read the binding patch (K)")
    results.report("target_values_kept", paper=99.8, label="binding patch, target's values kept (T)")
    results.save()


if __name__ == "__main__":
    main()
