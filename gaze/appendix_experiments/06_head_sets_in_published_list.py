"""Every head set of the paper against the published Gaze Heads (Appendix F.2, Table 15).

No model is needed. For each head set we count how many of its heads are in the top 100 and the
top 200 of the published Gaze Heads ranking, and in the top 100 of the Gaze score measured again on
our task (HEADS["gaze_score_on_this_task"]). The last column is the number expected by chance in a
top 100: the set's size times 100 / 1,152.

    python gaze/appendix_experiments/06_head_sets_in_published_list.py

Paper (published top 100 / published top 200 / re-measured top 100):
    binding, query and key heads (18)                     3 / 6 / 6
    relay heads (16)                                      4 / 5 / 3
    gaze heads (32)                                      23 / 29 / 29
    language heads (16)                                   0 / 0 / 0
    list task: outputs at the shape token (32)            5 / 11 / 13
    list task: image keys at the answer token (32)        8 / 20 / 19
    list task: queries at the last comma token (16)      11 / 15 / 14
    list task: outputs at the last letter, stop (64)     11 / 18 / 12
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.experiment import Results, parse_args
from utils.heads import HEADS

SETS = {"binding, query and key heads": ("binding_query_key", (3, 6, 6)), "relay heads": ("relay", (4, 5, 3)),
        "gaze heads": ("gaze", (23, 29, 29)), "language heads": ("language", (0, 0, 0)),
        "list task: outputs at the shape token": ("list_shape_token", (5, 11, 13)),
        "list task: image keys at the answer token": ("answer_token", (8, 20, 19)),
        "list task: queries at the last comma token": ("list_last_comma_queries", (11, 15, 14)),
        "list task: outputs at the last letter (stop)": ("stop", (11, 18, 12))}


def main():
    args = parse_args(__doc__)
    results = Results("head_sets_in_published_list", args)
    published, measured = HEADS["published_gaze_ranking"], HEADS["gaze_score_on_this_task"]
    print(f"{'head set':48s} heads layers   published 100 / 200   re-measured 100   by chance   paper")
    for label, (key, paper) in SETS.items():
        heads = set(HEADS[key])
        counts = (len(heads & set(published[:100])), len(heads & set(published[:200])), len(heads & set(measured[:100])))
        layers = f"{min(l for l, _ in heads)}-{max(l for l, _ in heads)}"
        print(f"{label:48s} {len(heads):5d} {layers:>6s}   {counts[0]:8d} / {counts[1]:3d}   {counts[2]:15d}"
              f"   {len(heads) * 100 / 1152:9.1f}   {paper[0]} / {paper[1]} / {paper[2]}")
        results.add(key, size=len(heads), published_top100=counts[0], published_top200=counts[1], measured_top100=counts[2])
    results.save()


if __name__ == "__main__":
    main()
