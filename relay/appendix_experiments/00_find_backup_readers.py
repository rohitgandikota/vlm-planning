"""Search for heads that read the letter when the gaze heads are blocked (Appendix C.3, part 1 of 2).

When the 32 gaze heads cannot see the image, most answers survive. Another set of heads might read
the letter in their place. To look for them we block the 32 gaze heads from the image at the colon
token, and run the gaze-head search again (gaze/main_experiments/00_find_gaze_heads.py) under this
block: one head at a time takes its query at the colon token from the source run. A head that
reads the letter in place of the gaze heads makes the target answer K, its own letter at the panel
the source points to. The gaze heads themselves are left out of the ranking.
    1. On the search pairs, score each head by how much its patch raises the logit of K over T.
    2. On the validation pairs, patch the top K heads together, for K = 1, 2, 4, ..., 256.
    3. Take the smallest K for which the target answers K in at least 80% of cases (the best K if
       none does).
The ranking is saved, and 05_backup_readers.py tests its top heads on the held-out pairs.

    python relay/appendix_experiments/00_find_backup_readers.py

--num_pairs is the number of search pairs (the paper uses 50), and half as many validation pairs
are used. To run step 1 in parts, give each part --start and --num_pairs, then run once without --start.

Paper: no number of heads moves the answer: 0.0% on the validation pairs at every K.
These numbers come from an earlier, smaller run (50 search pairs, 25 validation pairs of an earlier
pair set) with an earlier gaze-head list that shares 31 of its 32 heads with HEADS["gaze"].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, K_VALUES, choose_k, rank_heads, search_heads, sweep_k
from utils.model import load_model
from utils.patching import Patch, make_patch

SIZES = K_VALUES + (256,)


def queries_with_gaze_blocked(source, target, heads):
    """The heads take their query at the colon token from the source, while the 32 gaze heads
    cannot see the image at the colon token."""
    patch = make_patch(source, target, "query", "colon", heads=heads)
    return patch.add(Patch().block(HEADS["gaze"], [target.token("colon")], target.image_positions))


def main():
    args = parse_args(__doc__, split="search", num_pairs=50)
    model = load_model(args.model, args.device)
    results = Results("find_backup_readers", args)
    search_pairs = one_answer_pairs(model, "search", args.start + args.num_pairs)[args.start:]
    # The score is measured against the unpatched target, not against the target with the gaze
    # heads blocked. The difference is the same for every head of a case, so the ranking is the same.
    scores = search_heads(model, search_pairs, {"backup": (queries_with_gaze_blocked, "K")},
                          folder=Path(args.output) / "find_backup_readers_scores")["backup"]
    if args.start:                                                  # one part of step 1: nothing to select yet
        return
    ranking = [head for head in rank_heads(scores) if head not in HEADS["gaze"]][:max(SIZES)]
    validation_pairs = one_answer_pairs(model, "validation", max(1, args.num_pairs // 2))
    rates = sweep_k(model, validation_pairs, ranking, queries_with_gaze_blocked, "K", sizes=SIZES)
    k = choose_k({size: rate["letter"] for size, rate in rates.items()})
    print("K sweep with the gaze heads blocked: percent of validation cases that answer K")
    for size, rate in rates.items():
        print(f"  K = {size:3d}   {100 * rate['letter']:5.1f}" + ("   <- chosen" if size == k else ""))
    print("paper: 0.0 at every K")
    results.save(found={"ranking": ranking, "k_sweep": rates, "chosen_k": k, "scores": scores.tolist()})


if __name__ == "__main__":
    main()
