"""Which tokens carry the matching panels on the list task (Appendix E.3).

The list-task version of binding/main_experiments/02_residual_patch_every_token.py. We copy the
source's residual vector at one text token and one layer into the target, for every text token
after the image and every layer, and read the target's next letter. Three cases:
    first      nothing written; the source "source_first" has the shape at r2 and r4. Effect: the
               target answers its r2 letter (the first matching panel moved).
    next       W written; the standard source has the shape at r2 and r4 and wrote W at r2. Effect:
               the target answers its r3 letter (progress moved).
    next_set   W written; the source "source_set" has the shape at r1 and r3 and wrote W at r1, so
               only its set of matching panels differs. Effect: the target answers its r3 letter.

    python progress/appendix_experiments/48_residual_patch_every_token_list.py --num_pairs 500

Paper: on the first answer the map finds the same two tokens as on the single-answer task, the
shape token and the colon token. With the moved set, the residual patch at the shape token moves
the answer in 14.8% of cases (layer 20), and at the comma token in 78.4% (its best layer, 23).
The 14.8% is from the 500 held-out pairs. The map itself and the 78.4% come from an earlier,
smaller run (200 held-out pairs of an earlier draw of the list task, 194 valid). A case counts when
the target gives its own next letter with no patch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model, n_layers
from utils.patching import make_patch, run_patches

# case: (source image, letters written?, the target's matching panel the effect answers from: 1 is r1)
CASES = {"first": ("source_first", False, 2), "next": ("source", True, 3), "next_set": ("source_set", True, 3)}


def outcome(answer, target_side, source_side):
    """r1 to r4 (the target's letters at its matching panels), source_letter, end, or other."""
    letters = matching_letters(target_side)
    if answer.strip() in letters:
        return f"r{letters.index(answer.strip()) + 1}"
    if answer.strip() in source_side["letters"] and answer.strip() not in target_side["letters"]:
        return "source_letter"
    return "end" if answer == "<end>" else "other"


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    boundaries = range(n_layers(model) + 1)                        # the last one is after the last layer
    results = Results("residual_patch_every_token_list", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for name, (side, written, effect) in CASES.items():
            written = [pair["written"]] if written else []
            target = run_list(model, pair["target"], written)
            if target.answer != " " + matching_letters(pair["target"])[len(written)]:
                continue
            source = run_list(model, pair[side], written)
            tokens = [position for position, _ in target.text_tokens()]
            patches = [make_patch(source, target, "residual", t, layer=b) for t in tokens for b in boundaries]
            said = [outcome(a, pair["target"], pair[side]) for a in run_patches(model, target, patches, max_batch=32)["answers"]]
            grid = [said[i * len(boundaries):(i + 1) * len(boundaries)] for i in range(len(tokens))]   # tokens x layers
            named = {"shape": target.token("shape"), "colon": target.token("colon"), "comma": target.last}
            row = {f"{name}_{token}_layer_{b:02d}": grid[tokens.index(position)][b] == f"r{effect}"
                   for token, position in named.items() for b in boundaries}
            results.add(pair["index"], case=name, tokens=[text for _, text in target.text_tokens()], grid=grid, **row)
    for name, token, layer, paper in (("first", "shape", None, None), ("first", "colon", None, None),
                                      ("next_set", "shape", 20, 14.8), ("next_set", "comma", None, 78.4)):
        rates = {b: results.rate(f"{name}_{token}_layer_{b:02d}")["percent"] or 0 for b in boundaries}
        best = max(rates, key=rates.get) if layer is None else layer
        results.report(f"{name}_{token}_layer_{best:02d}", paper=paper, label=f"{name}, {token} token, layer {best}")
    results.save()


if __name__ == "__main__":
    main()
