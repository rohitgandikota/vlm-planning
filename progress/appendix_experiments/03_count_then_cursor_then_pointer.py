"""Count, then cursor, then the source's next pointer (Appendix D.2, figure panel c).

The target has its shape at r1 to r4 and has written nothing. The source is the standard source of
the list task: its shape is at r2 and r4 only, and it has written its first letter, the target's
letter W, which it holds at r2. We patch the residual vector of the source's comma token into the
target's colon token, one layer at a time from 14 to 32. Each account predicts another answer:
    r2       a count: "one letter is done", so the target skips one panel
    r3       a cursor: "done up to r2", so the target starts after the written panel
    r4       the source's own next pointer (its next matching panel is r4)
    source   the source's next letter itself

    python progress/appendix_experiments/03_count_then_cursor_then_pointer.py --num_pairs 500

Paper: r2 in 86.5% of cases at layer 19; r3 in 68.4% at layer 20 and 53.9% at layer 21; r4 in
77.7% at layer 22 and 87.6% at layer 24; the source's letter in 99.5% at layer 30.
These numbers came from an earlier, smaller run: 200 list pairs drawn from another seed, of which
193 count. Every held-out pair here counts (both runs give their own next letter with no patch).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, matching_letters, name_answer, run_list
from utils.experiment import Results, parse_args
from utils.model import load_model
from utils.patching import make_patch, run_patches

LAYERS = range(14, 33)
REPORT = [(19, "r2", 86.5), (20, "r3", 68.4), (21, "r3", 53.9), (22, "r4", 77.7), (24, "r4", 87.6), (30, "source", 99.5)]


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("count_then_cursor_then_pointer", args)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        names = {f"r{i + 1}": letter for i, letter in enumerate(matching_letters(pair["target"]))}
        names["source"] = matching_letters(pair["source"])[1]           # the source's letter at r4
        target = run_list(model, pair["target"], [])
        source = run_list(model, pair["source"], [pair["written"]])
        patches = [make_patch(source, target, "residual", "colon", source_token="last_comma", layer=layer)
                   for layer in LAYERS]
        said = [name_answer(answer, names) for answer in run_patches(model, target, patches)["answers"]]
        results.add(pair["index"], said=said,
                    **{f"layer_{layer}_says_{name}": panel == name for layer, panel in zip(LAYERS, said) for name in names})
    for layer, name, paper in REPORT:
        results.report(f"layer_{layer}_says_{name}", paper=paper, label=f"layer {layer}: answers {name}")
    results.save()


if __name__ == "__main__":
    main()
