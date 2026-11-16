"""Without the relay heads the pointer does not reach the colon token (Section 4.2).

No source run is used here. In the target's own run, we block the attention of the 16 relay heads
from the colon token to the shape token and the tokens after it. If the relay heads are needed to
bring the pointer to the colon token, the target can no longer give its own answer (outcome T).

    python relay/main_experiments/03_relay_heads_blocked.py --num_pairs 500

Paper: the target keeps its answer in only 7.8% of cases. When the same number of random heads
from the same layers are blocked, it keeps it in 99.8%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import Patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("relay_heads_blocked", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = random_like(HEADS["relay"], seed=1000 + pair["index"])
        for _, target, letters in both_directions(model, pair):     # each side of the pair is a target once
            colon = target.token("colon")
            blocked = range(target.token("shape"), colon)           # the shape token through the token before the colon token
            patches = [Patch().block(HEADS["relay"], [colon], blocked),
                       Patch().block(random_heads, [colon], blocked)]
            relay, control = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], relay_heads_blocked=relay == " " + letters["T"],
                        random_heads_blocked=control == " " + letters["T"], answer=relay)
    results.report("relay_heads_blocked", paper=7.8, label="relay heads blocked: own answer kept (T)")
    results.report("random_heads_blocked", paper=99.8, label="random heads blocked: own answer kept (T)")
    results.save()


if __name__ == "__main__":
    main()
