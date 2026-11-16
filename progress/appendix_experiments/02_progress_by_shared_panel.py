"""The early skip is tied to panel 1 (Appendix D.2, figure panel b).

This splits the result of progress/main_experiments/02_progress_at_the_comma_token.py by the shared
panel: the source's written panel, which is also the target's first matching panel (panel 1, 2 or
3). It runs no model. It reads the rows that the main experiment saved, so run that file first
with the same --output folder. For each group we report how often the target gives its second
answer, averaged over layers 0 to 17 (the early layers), and at layers 20 and 21.

    python progress/main_experiments/02_progress_at_the_comma_token.py --num_pairs 500
    python progress/appendix_experiments/02_progress_by_shared_panel.py

Paper: in the early layers the target skips its first matching panel in 80.6% of pairs when the
shared panel is panel 1, 8.6% for panel 2, and 4.7% for panel 3. At layer 20 the three groups skip
in 99.5%, 97.5%, and 74.2%, and at layer 21 in 53.2%, 72.0%, and 88.4%.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.experiment import Results, interval, parse_args

EARLY = range(18)                                    # layers 0 to 17
PAPER = {0: (80.6, 99.5, 53.2), 1: (8.6, 97.5, 72.0), 2: (4.7, 74.2, 88.4)}


def main():
    args = parse_args(__doc__)
    saved = Path(args.output) / "progress_at_the_comma_token.json"
    if not saved.exists():
        sys.exit(f"{saved} not found. Run progress/main_experiments/02_progress_at_the_comma_token.py first.")
    rows = json.loads(saved.read_text())["rows"]
    results = Results("progress_by_shared_panel", args)
    for row in rows:
        panel = row["shared_panel"]
        results.add(row["pair"], shared_panel=panel,
                    early_share=sum(row["outcomes"][layer] == "second" for layer in EARLY) / len(EARLY),
                    **{f"panel_{p + 1}_layer_{layer}": row[f"layer_{layer}"] if p == panel else None
                       for p in range(3) for layer in (20, 21)})
    for panel, (early, at_20, at_21) in PAPER.items():
        group = [r for r in results.rows if r["shared_panel"] == panel]
        if not group:                                      # a small --num_pairs may leave a panel with no pair
            continue
        shares = [r["early_share"] for r in group]
        low, high = interval(shares, [1.0] * len(group), [r["pair"] for r in group])
        print(f"{f'shared panel {panel + 1}: second answer, layers 0-17':45s} {100 * sum(shares) / max(len(group), 1):5.1f}%  "
              f"[{low:.1f}, {high:.1f}]  ({len(group)} pairs)   paper: {early}")
        results.report(f"panel_{panel + 1}_layer_20", paper=at_20, label=f"shared panel {panel + 1}: second answer, layer 20")
        results.report(f"panel_{panel + 1}_layer_21", paper=at_21, label=f"shared panel {panel + 1}: second answer, layer 21")
    results.save()


if __name__ == "__main__":
    main()
