"""The relay heads at every model size (Appendix G.1.2, Table "sizes-one", relay rows).

As in relay/main_experiments/04_relay_reads_binding_patch.py, with each size's own heads.
    1. The binding patch: the target with the outputs of the binding heads patched from the source
       at the shape token. We record this patched run.
    2. An unpatched target in which the relay heads at the colon token read their values, from the
       shape token on, out of run 1. If they carry the pointer, the target answers K.
The control in the other direction keeps the binding patch but lets the relay heads read the
target's own values: the target's own answer T should come back. Random heads from the same layers
(seed = pair index) replace the relay heads in run 2 as the matched control.

    python generalization/main_experiments/03_sizes_relay_heads.py --model Qwen/Qwen3-VL-2B-Instruct --num_pairs 500

Models: any Qwen3-VL size. 2B 16 GB, 4B 24 GB, 8B 48 GB, 32B 160 GB (one B200) or --device cpu.
1 recorded and 3 patched runs per case, 2 cases per pair.

Paper (2B / 4B / 8B / 32B; the 2B and 4B on 100 earlier held-out pairs, the 32B on 50): relay values
84.5 / 95.5 / 89.0 / 84.0; target's values restored 91.0 / 98.0 / 99.8 / 100.0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, pairs_of_8b
from utils.experiment import Results, parse_args
from utils.heads import heads_of_size, random_heads_like
from utils.model import load_model
from utils.patching import make_patch, record_patch, run_patches

PAPER = {"relay_heads": {"2B": 84.5, "4B": 95.5, "8B": 89.0, "32B": 84.0},
         "target_values_kept": {"2B": 91.0, "4B": 98.0, "8B": 99.8, "32B": 100.0}}


def relay_values(run, target, heads):
    """At the colon token of the target, the heads read the values of `run` from the shape token on."""
    return make_patch(run, target, "values", "colon", heads=heads, positions=range(target.token("shape"), target.token("colon")))


def main():
    args = parse_args(__doc__)
    pairs = pairs_of_8b("one_answer", args.split, args.start + args.num_pairs, args.device)[args.start:]
    model = load_model(args.model, args.device)
    binding, relay = heads_of_size(args.model, "binding"), heads_of_size(args.model, "relay")
    results = Results(f"sizes_relay_heads_{args.model.split('/')[-1]}", args)
    for pair in pairs:
        random_heads = random_heads_like(model, relay, seed=pair["index"])
        for source, target, letters in both_directions(model, pair):
            binding_patch = make_patch(source, target, "head_output", "shape", heads=binding)
            patched = record_patch(model, target, binding_patch)                                  # run 1
            patches = [relay_values(patched, target, relay),                                      # run 2
                       relay_values(patched, target, random_heads),
                       binding_patch.add(relay_values(target, target, relay))]                    # values kept
            moved, control, kept = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], relay_heads=moved == " " + letters["K"], random_heads=control == " " + letters["K"],
                        target_values_kept=kept == " " + letters["T"], answer=moved)
    size = args.model.split("-")[-2]
    results.report("relay_heads", paper=PAPER["relay_heads"].get(size), label="relay heads read the binding patch (K)")
    results.report("random_heads", label="random heads read the binding patch (K)")
    results.report("target_values_kept", paper=PAPER["target_values_kept"].get(size), label="binding patch, target's values kept (T)")
    results.save()


if __name__ == "__main__":
    main()
