"""Small helpers shared by the experiment scripts: arguments, counting, intervals, saving.

    args = parse_args(__doc__)
    results = Results("binding_heads", args)
    results.add(pair_index, binding=True, random=False)       # one case: did each patch give the effect?
    results.report("binding", paper=85.9)                     # prints the rate and its 95% interval
    results.save()
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args(description, **defaults):
    """The arguments every experiment takes. `defaults` overrides a default, for example num_pairs=100."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_pairs", type=int, default=defaults.get("num_pairs", 500),
                        help="how many source-target pairs to run (the paper uses 500 held-out pairs)")
    parser.add_argument("--split", default=defaults.get("split", "heldout"), choices=["search", "validation", "heldout"])
    parser.add_argument("--model", default=defaults.get("model", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=str(ROOT / "results"), help="folder for the result files")
    parser.add_argument("--start", type=int, default=0, help="skip this many pairs of the split (to run in parts)")
    return parser.parse_args()


def interval(hits, cases, groups, resamples=2000, seed=0):
    """95% bootstrap interval of sum(hits) / sum(cases), resampling whole pairs. Returns (low, high) in percent."""
    ids = sorted(set(groups))
    row = {g: i for i, g in enumerate(ids)}
    h, c = np.zeros(len(ids)), np.zeros(len(ids))
    for hit, case, group in zip(hits, cases, groups):
        h[row[group]] += hit
        c[row[group]] += case
    weights = np.random.default_rng(seed).multinomial(len(ids), np.ones(len(ids)) / len(ids), size=resamples)
    rates = (weights @ h) / np.maximum(weights @ c, 1)
    return float(np.percentile(rates, 2.5) * 100), float(np.percentile(rates, 97.5) * 100)


class Results:
    """Counts, for each named condition, in how many cases the effect happened."""

    def __init__(self, name, args=None):
        self.name, self.args = name, args
        self.rows = []                                   # one dict per case
        self.reported = []                               # what `report` printed, saved with the results

    def add(self, pair, **conditions):
        """One case of one pair. Each condition is True, False, or None (the case does not count for it).
        Any other value (a string, a number) is saved with the row but not counted."""
        self.rows.append({"pair": pair, **conditions})

    def rate(self, condition):
        rows = [r for r in self.rows if isinstance(r.get(condition), bool)]
        hits = [float(r[condition]) for r in rows]
        if not rows:
            return {"percent": None, "cases": 0}
        low, high = interval(hits, [1.0] * len(rows), [r["pair"] for r in rows])
        return {"percent": 100 * sum(hits) / len(rows), "low": low, "high": high, "cases": len(rows)}

    def report(self, condition, paper=None, label=None):
        r = self.rate(condition)
        text = f"{label or condition:45s} "
        text += "no cases" if r["percent"] is None else f"{r['percent']:5.1f}%  [{r['low']:.1f}, {r['high']:.1f}]  ({r['cases']} cases)"
        if paper is not None:
            text += f"   paper: {paper}"
        print(text, flush=True)
        self.reported.append({"condition": condition, "label": label or condition, "paper": paper, **r})
        return r

    def save(self, **extra):
        folder = Path(self.args.output if self.args else ROOT / "results")
        folder.mkdir(parents=True, exist_ok=True)
        conditions = sorted({k for r in self.rows for k, v in r.items() if isinstance(v, bool)})
        suffix = f"_from{self.args.start}" if self.args and self.args.start else ""
        path = folder / f"{self.name}{suffix}.json"
        path.write_text(json.dumps({"experiment": self.name, "args": vars(self.args) if self.args else {},
                                    "rates": {c: self.rate(c) for c in conditions}, "reported": self.reported,
                                    "rows": self.rows, **extra}, indent=1))
        print(f"saved {path}", flush=True)


def run_folder(folder):
    """Run every experiment file of one mechanism folder, in order. Used by the `run_<folder>.py` files.

    Runs `main_experiments/*.py`, and `appendix_experiments/*.py` too with `--appendix`. Files that
    start with `00_` are the head searches; they are slow and are skipped unless `--search` is given.
    Every other argument is passed on to each experiment (for example `--num_pairs 100`)."""
    import subprocess
    import sys
    args = sys.argv[1:]
    appendix, search = "--appendix" in args, "--search" in args
    args = [a for a in args if a not in ("--appendix", "--search")]
    files = sorted((Path(folder) / "main_experiments").glob("[0-9]*.py"))
    if appendix:
        files += sorted((Path(folder) / "appendix_experiments").glob("[0-9]*.py"))
    failed = []
    for file in files:
        if file.name.startswith("00_") and not search:
            continue
        print(f"\n===== {file.relative_to(Path(folder).parent)} =====", flush=True)
        if subprocess.run([sys.executable, str(file), *args]).returncode != 0:
            failed.append(file.name)
    print("\nall experiments finished" if not failed else f"\nfailed: {failed}", flush=True)
