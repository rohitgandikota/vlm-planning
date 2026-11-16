"""Blocking experiments at every model size (Appendix G.1.2, Table "sizes-one", blocking rows).

No source run. In the target's own run we block attention and ask whether the target keeps its own
answer (outcome T):
    nothing blocked
    binding heads blocked from the image at the shape token. On the 8B these are the binding, query
        and key heads (18); on the other sizes the paper's runs blocked the query and key heads.
    later tokens cannot attend to the shape token (every head of every layer)
    relay heads cannot attend to the text at the colon token (the shape token through the token
        before the colon token), and the same with random heads from the same layers
Random heads use the seed 1000 + pair index, as in relay/main_experiments/03_relay_heads_blocked.py.

    python generalization/main_experiments/04_sizes_blocking.py --model Qwen/Qwen3-VL-2B-Instruct --num_pairs 500

Models: any Qwen3-VL size. 2B 16 GB, 4B 24 GB, 8B 48 GB, 32B 160 GB (one B200) or --device cpu.
6 patched runs per case, 2 cases per pair.

Paper (2B / 4B / 8B / 32B; the 2B and 4B on 100 earlier held-out pairs, the 32B on 50): nothing blocked
93.0 / 97.0 / 100.0 / 100.0; binding heads blocked 12.0 / 57.5 / 84.9 / 100.0; shape token not read
10.5 / 24.5 / 25.3 / 31.0; relay heads blocked 3.0 / 28.0 / 7.8 / 100.0; random heads 93.5 / 97.0 / 99.8 / 100.0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, pairs_of_8b
from utils.experiment import Results, parse_args
from utils.heads import heads_of_size, random_heads_like
from utils.model import load_model, n_layers
from utils.patching import Patch, run_patches

PAPER = {"nothing_blocked": {"2B": 93.0, "4B": 97.0, "8B": 100.0, "32B": 100.0},
         "binding_heads_blocked": {"2B": 12.0, "4B": 57.5, "8B": 84.9, "32B": 100.0},
         "shape_token_not_read": {"2B": 10.5, "4B": 24.5, "8B": 25.3, "32B": 31.0},
         "relay_heads_blocked": {"2B": 3.0, "4B": 28.0, "8B": 7.8, "32B": 100.0},
         "random_heads_blocked": {"2B": 93.5, "4B": 97.0, "8B": 99.8, "32B": 100.0}}


def main():
    args = parse_args(__doc__)
    pairs = pairs_of_8b("one_answer", args.split, args.start + args.num_pairs, args.device)[args.start:]
    model = load_model(args.model, args.device)
    binding, relay = heads_of_size(args.model, "blocked_at_shape_token"), heads_of_size(args.model, "relay")
    every_layer = range(n_layers(model))
    results = Results(f"sizes_blocking_{args.model.split('/')[-1]}", args)
    for pair in pairs:
        random_relay = random_heads_like(model, relay, seed=1000 + pair["index"])
        for _, target, letters in both_directions(model, pair):          # each side of the pair is a target once
            shape, colon, image = target.token("shape"), target.token("colon"), target.image_positions
            text = range(shape, colon)                                     # the shape token through the token before the colon
            patches = {"nothing_blocked": Patch(),
                       "binding_heads_blocked": Patch().block(binding, [shape], image),
                       "shape_token_not_read": Patch().block(None, range(shape + 1, colon + 1), [shape], layers=every_layer),
                       "relay_heads_blocked": Patch().block(relay, [colon], text),
                       "random_heads_blocked": Patch().block(random_relay, [colon], text)}
            answers = run_patches(model, target, list(patches.values()))["answers"]
            results.add(pair["index"], **{name: answer == " " + letters["T"] for name, answer in zip(patches, answers)})
    size = args.model.split("-")[-2]
    for name in PAPER:
        results.report(name, paper=PAPER[name].get(size), label=f"{name.replace('_', ' ')}: own answer kept (T)")
    results.save()


if __name__ == "__main__":
    main()
