"""Where the binding and gaze heads look, layer by layer, with no patch (Appendix E.4, "Attention by layer").

No patch. Two readouts of the recorded attention:
    by layer      with W written, the share of each head's attention at the last comma token that
                  falls on the written panel (r1), the next panel (r2), each other matching panel
                  (r3, r4, averaged) and each panel without the shape (averaged), for the binding
                  heads and for the gaze heads, averaged over the heads of each layer. As in
                  progress/main_experiments/12, the targets are drawn again with every shape at its
                  panel center, and a share is of the head's whole attention, text tokens included.
    chosen panel  on the stored images, the attention of the binding heads at the shape token, and
                  of the gaze heads at the colon token (first answer) and at the comma token (second
                  answer), summed over the heads and rescaled to the image: the share on the matching
                  panels that are still open, and of that, the share on the panel answered next.

    python progress/appendix_experiments/51_attention_at_the_comma_token.py --num_pairs 500

Paper: at layer 21 the binding heads put 51.6% on the next panel and 3.9% on the written panel. At
layer 20 the gaze heads give 22.3% to the next panel and 23.0% to each other matching panel; by layer
23 these shares are 61.9% and 7.4%. At the shape token the binding heads put 79.6% of their image
attention on the matching panels and 27.4% of that on the one chosen first. The gaze heads put
43.8% (first answer) and 65.9% (second answer) of their attention to open matching panels on the
chosen one. The binding heads here are the 16 binding heads with the query and key heads (18 heads,
HEADS["binding_query_key"]), as in the run that produced these numbers. Every held-out pair is used.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import list_pairs, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import attention_share, report_mean

SETS = {"binding": HEADS["binding_query_key"], "gaze": HEADS["gaze"]}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("attention_at_the_comma_token", args)
    by_layer, chosen = {}, {}                                      # name -> list of (pair, value)
    for pair in list_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        r = pair["panels"]
        fillers = [p for p in range(6) if p not in r]
        # By layer, on the centered image, at the last comma token.
        run = run_list(model, {**pair["target"], "centers": None}, [pair["written"]])
        for name, heads in SETS.items():
            for layer in sorted({l for l, _ in heads}):
                share = attention_share(run, [h for h in heads if h[0] == layer], "last_comma").mean(0)
                for role, value in (("written", share[r[0]]), ("next", share[r[1]]),
                                    ("other", share[r[2:]].mean()), ("fillers", share[fillers].mean())):
                    by_layer.setdefault(f"{name}|{layer}|{role}", []).append((pair["index"], float(value)))
        # The chosen panel, on the stored image. The image shares are summed over the heads. Only the
        # pairs whose first and second answers are right with no patch count here.
        first, second = run_list(model, pair["target"], []), run_list(model, pair["target"], [pair["written"]])
        letters = [pair["target"]["letters"][p] for p in r]
        if first.answer != " " + letters[0] or second.answer != " " + letters[1]:
            continue
        for name, run_, heads, token, open_panels in (("shape_token_binding", first, SETS["binding"], "shape", r),
                                                      ("first_answer_gaze", first, SETS["gaze"], "colon", r),
                                                      ("second_answer_gaze", second, SETS["gaze"], "last_comma", r[1:])):
            mass = attention_share(run_, heads, token).sum(0)       # per panel, summed over the heads
            on_open = float(mass[open_panels].sum())
            chosen.setdefault(f"{name}|on_matching", []).append((pair["index"], on_open, float(mass.sum())))
            chosen.setdefault(f"{name}|of_that_on_chosen", []).append((pair["index"], float(mass[open_panels[0]]), on_open))
        results.add(pair["index"], first_answer=first.answer, second_answer=second.answer)
    results.save(by_layer={k: sum(v for _, v in rows) / len(rows) for k, rows in by_layer.items()},
                 chosen={k: sum(a for _, a, _ in rows) / sum(b for _, _, b in rows) for k, rows in chosen.items()})
    for key, paper in (("binding|21|next", 51.6), ("binding|21|written", 3.9), ("gaze|20|next", 22.3), ("gaze|20|other", 23.0),
                       ("gaze|23|next", 61.9), ("gaze|23|other", 7.4)):
        rows = by_layer[key]
        report_mean(f"comma token, {key}", [v for _, v in rows], [p for p, _ in rows], paper=paper)
    for key, paper in (("shape_token_binding|on_matching", 79.6), ("shape_token_binding|of_that_on_chosen", 27.4),
                       ("first_answer_gaze|of_that_on_chosen", 43.8), ("second_answer_gaze|of_that_on_chosen", 65.9)):
        rows = chosen[key]                                         # a ratio of sums over the pairs, as in the paper's run
        report_mean(key, [a for _, a, _ in rows], [p for p, _, _ in rows], paper=paper, totals=[b for _, _, b in rows])


if __name__ == "__main__":
    main()
