"""What looks like a count is the record of earlier panels (Appendix D.4, last paragraph).

The target has written only its letter at r3, so the panels r1 and r2 were never written. On its
own it answers either r2 (going back) or r4 (going on); progress/main_experiments/08_letters_out_of_order.py
counts this (34.4% r2). The source is a run on the same image that wrote r1, r2 and r3. Two
patches carry the source's record into the target:
    comma residual   the residual vector of the source's last comma token replaces the target's,
                     at one layer (16, 18, 20, 21, 22, 24, 26)
    letter values    at the target's comma token, a set of heads reads the source's values of its
                     third answer token in place of the target's only answer token (all heads, or
                     the 16 relay heads)
If the last answer token carries the earlier panels, the target answers r4.

    python progress/appendix_experiments/12_record_of_earlier_panels.py --num_pairs 500

Paper: the comma residual at layer 22 gives r4 in 98.5% of cases. The relay heads reading the third
letter's values give it in 89.3% (all heads 81.2%).
These numbers came from an earlier, smaller run: 200 list pairs drawn from another seed, of which
197 count. A pair counts when the source, with r1, r2 and r3 written, answers r4 with no patch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import all_heads, load_model
from utils.patching import make_patch, run_patches

LAYERS = (16, 18, 20, 21, 22, 24, 26)


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("record_of_earlier_panels", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        letters = matching_letters(pair["target"])
        target = run_list(model, pair["target"], [letters[2]])                  # r3 only
        source = run_list(model, pair["target"], letters[:3])                   # r1, r2, r3
        if source.answer != " " + letters[3]:
            continue
        comma = target.token("last_comma")
        values = lambda heads: make_patch(source, target, "values", comma, source_token="last_comma", heads=heads,
                                          positions=[target.token("last_answer")], source_positions=[source.token("last_answer")])
        patches = [make_patch(source, target, "residual", comma, source_token="last_comma", layer=layer) for layer in LAYERS]
        patches += [values(all_heads(model)), values(HEADS["relay"])]
        answers = run_patches(model, target, patches)["answers"]
        says_r4 = [answer == " " + letters[3] for answer in answers]
        results.add(pair["index"], **{f"residual_layer_{layer}": hit for layer, hit in zip(LAYERS, says_r4)},
                    all_heads_values=says_r4[-2], relay_heads_values=says_r4[-1])
    for layer in LAYERS:
        results.report(f"residual_layer_{layer}", paper=98.5 if layer == 22 else None, label=f"comma residual at layer {layer}: answers r4")
    results.report("all_heads_values", paper=81.2, label="all heads read the third letter's values: r4")
    results.report("relay_heads_values", paper=89.3, label="relay heads read the third letter's values: r4")
    results.save()


if __name__ == "__main__":
    main()
