"""The heads found with the gaze heads blocked do not read the letter (Appendix C.3, part 2 of 2).

Run 00_find_backup_readers.py first, with the same --output: this file reads its ranking. On the
held-out pairs, with the 32 gaze heads blocked from the image at the colon token, we test the top
32 and the top 128 heads of that ranking in two ways.
    query patch   they take their queries at the colon token from the source. If they read the
                  letter, the target answers K, its own letter at the panel the source points to.
    blocked       they are blocked from the image at the colon token as well. If they are the
                  backup, the target should lose its own answer T.
The controls are the same number of random heads from the same layers, never gaze heads. For
reference we also run the 32 gaze heads blocked alone (T) and their query patch with nothing
blocked (K).

    python relay/appendix_experiments/05_backup_readers.py --num_pairs 500

Paper: with the gaze heads blocked, the query patch gives K in 3.1% (top 32) and 5.6% (top 128),
against 74.0% for the query patch of the 32 gaze heads with nothing blocked. Blocking the top heads
too keeps T in 86.2% (top 32) and 82.1% (top 128), against 88.2% with the gaze heads blocked alone;
random heads keep it in 87.8% and 92.9%.
The 3.1, 5.6, 86.2, 82.1, 87.8 and 92.9 come from an earlier, smaller run (196 cases of 100 pairs of
an earlier pair set) with an earlier gaze-head list; the 74.0 and 88.2 are on the 500 held-out pairs.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import Patch, make_patch, run_patches

PAPER = {"gaze_blocked": 88.2, "gaze_query_patch": 74.0,
         "top32_query_patch": 3.1, "top32_blocked_too": 86.2, "random32_blocked_too": 87.8,
         "top128_query_patch": 5.6, "top128_blocked_too": 82.1, "random128_blocked_too": 92.9}


def main():
    args = parse_args(__doc__)
    found_file = Path(args.output) / "find_backup_readers.json"
    if not found_file.exists():
        sys.exit(f"{found_file} not found. Run relay/appendix_experiments/00_find_backup_readers.py first.")
    ranking = [tuple(head) for head in json.loads(found_file.read_text())["found"]["ranking"]]
    model = load_model(args.model, args.device)
    results = Results("backup_readers", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        found = {k: ranking[:k] for k in (32, 128)}
        random = {k: random_like(heads, seed=pair["index"], exclude=HEADS["gaze"]) for k, heads in found.items()}
        for source, target, letters in both_directions(model, pair):
            colon, image = target.token("colon"), target.image_positions
            gaze_blocked = lambda: Patch().block(HEADS["gaze"], [colon], image)
            patches = {"gaze_blocked": gaze_blocked(),
                       "gaze_query_patch": make_patch(source, target, "query", "colon", heads=HEADS["gaze"])}
            for k in found:
                patches[f"top{k}_query_patch"] = make_patch(source, target, "query", "colon", heads=found[k]).add(gaze_blocked())
                patches[f"random{k}_query_patch"] = make_patch(source, target, "query", "colon", heads=random[k]).add(gaze_blocked())
                patches[f"top{k}_blocked_too"] = gaze_blocked().block(found[k], [colon], image)
                patches[f"random{k}_blocked_too"] = gaze_blocked().block(random[k], [colon], image)
            answers = dict(zip(patches, run_patches(model, target, list(patches.values()))["answers"]))
            results.add(pair["index"], **{name: answer == " " + letters["K" if "query_patch" in name else "T"]
                                          for name, answer in answers.items()})
    for name in ("gaze_blocked", "gaze_query_patch", "top32_query_patch", "random32_query_patch", "top32_blocked_too",
                 "random32_blocked_too", "top128_query_patch", "random128_query_patch", "top128_blocked_too", "random128_blocked_too"):
        outcome = "pointer moved (K)" if "query_patch" in name else "own answer kept (T)"
        results.report(name, paper=PAPER.get(name), label=f"{name}: {outcome}")
    results.save()


if __name__ == "__main__":
    main()
