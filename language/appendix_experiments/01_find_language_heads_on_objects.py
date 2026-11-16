"""Find the language heads on the object task (Appendix F.6).

The same search as language/main_experiments/00_find_language_heads.py, on the object task: for all
1,152 heads, one at a time, the head's output at the last prompt token is copied from a source that
asks in language A about image X into a target that asks in language B about image Y. The score is
the gain of the target's object in the source's language A over the target's object in its own
language B, read at the first token the model writes for each word. Heads are ranked by their mean
score over the 12 search images, their 3 sources each, and all 12 ordered pairs of languages (432
cases). We keep the top 16.

    python language/appendix_experiments/01_find_language_heads_on_objects.py

--num_pairs is the number of search images (the paper uses all 12). The object images need the
Noto Color Emoji font (see EMOJI_FONT in utils/dataset.py).

Paper: the 16 heads of HEADS["language_objects"]. The two tasks share 10 of their 16 heads, and their
two strongest heads are the same. The object search is an earlier run, and the 10 is counted
against an earlier colored-shapes search; against HEADS["language"] the count is 9.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import LANGUAGES, language_task
from utils.experiment import Results, parse_args
from utils.heads import HEADS, rank_heads, score_each_head
from utils.model import load_model


def main():
    args = parse_args(__doc__, split="search", num_pairs=12)
    model = load_model(args.model, args.device)
    task = language_task("objects")
    results = Results("find_language_heads_on_objects", args)
    total, cases = 0, 0
    for spec in task.images("search", args.start + args.num_pairs)[args.start:]:
        for source, target, case in task.cases(model, spec, target_languages=LANGUAGES):
            moved = task.word_token(model, case["source_language"], case["target_concept"])     # the language moved
            own = task.word_token(model, case["target_language"], case["target_concept"])       # nothing changed
            total += score_each_head(model, source, target, "head_output", "last", lambda logits: logits[moved] - logits[own])
            cases += 1
        print(f"image {spec['index']} searched ({cases} cases)", flush=True)
    ranking = rank_heads(total / cases)
    print(f"language heads found: {ranking[:16]}")
    print(f"same as the paper's object-task heads: {ranking[:16] == HEADS['language_objects']}")
    print(f"shared with the colored-shapes language heads: {len(set(ranking[:16]) & set(HEADS['language']))} of 16")
    results.save(found=ranking[:16], ranking=ranking, scores=(total / cases).tolist())


if __name__ == "__main__":
    main()
