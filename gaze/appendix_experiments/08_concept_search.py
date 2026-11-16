"""Are the gaze heads the heads that carry the concept on the language tasks? (Appendix F.4 and F.5)

The gaze heads were found on the letter task. To check whether the same heads carry the concept on
the colored-shapes task, we search again there. A source asks in language A about image X, and a
target asks in language B about image Y, which shows another concept. For all 1,152 heads, one at a
time, we copy the head's output at the last prompt token from the source into the target, and
score the gain of the source's concept over the target's concept, both as the first token of the
word in the target's language B. Each search image takes its first source, with all 12 ordered
pairs of languages (12 x 12 = 144 cases). Heads are ranked by their mean score.

    python gaze/appendix_experiments/08_concept_search.py
    python gaze/appendix_experiments/08_concept_search.py --task objects

--num_pairs is the number of search images (the paper uses all 12).

Paper, colored shapes: 14 of the 32 gaze heads are in the top 32 of this search, and 25 in its top
128 (the ranking is HEADS["concept_search"]). Objects: 14 in the top 32 and 28 in the top 128, and
21 of its top 32 are also in the top 32 of the colored-shapes search. The object numbers come from
an earlier run (HEADS["concept_search_objects"]), counted against an earlier list of gaze heads that
shares 31 of its 32 heads with HEADS["gaze"], and against an earlier colored-shapes search; with
HEADS["concept_search"] the last count is 20.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import LANGUAGES, task_argument
from utils.experiment import Results, parse_args
from utils.heads import HEADS, rank_heads, score_each_head
from utils.model import load_model


def main():
    task = task_argument()
    args = parse_args(__doc__, split="search", num_pairs=12)
    model = load_model(args.model, args.device)
    results = Results(f"concept_search_{task.name}", args)
    total, cases = 0, 0
    for spec in task.images("search", args.start + args.num_pairs)[args.start:]:
        source_spec = spec["sources"][0]
        targets = {language: task.run(model, spec, language) for language in LANGUAGES}
        sources = {language: task.run(model, source_spec, language) for language in LANGUAGES}
        for target_language in LANGUAGES:
            moved = task.word_token(model, target_language, task.concept(source_spec))     # the source's concept
            own = task.word_token(model, target_language, task.concept(spec))              # the target's concept
            for source_language in LANGUAGES:
                if source_language != target_language:
                    total += score_each_head(model, sources[source_language], targets[target_language], "head_output",
                                             "last", lambda logits: logits[moved] - logits[own])
                    cases += 1
        print(f"image {spec['index']} searched ({cases} cases)", flush=True)
    ranking = rank_heads(total / cases)
    for k in (16, 32, 64, 128):
        print(f"gaze heads in the top {k:3d} of this search: {len(set(HEADS['gaze']) & set(ranking[:k]))}")
    print("paper: 14 in the top 32 and 25 in the top 128 (colored shapes); 14 and 28 (objects)")
    if task.name == "objects":
        print(f"top 32 shared with the colored-shapes search: {len(set(ranking[:32]) & set(HEADS['concept_search'][:32]))} (paper: 21)")
    results.save(ranking=ranking[:256], scores=(total / cases).tolist(), cases=cases)


if __name__ == "__main__":
    main()
