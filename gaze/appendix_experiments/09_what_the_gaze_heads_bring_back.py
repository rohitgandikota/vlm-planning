"""What the gaze heads bring back, on the colored-shapes task and the object task (Appendix F.4, F.5).

This extends language/main_experiments/02 and 03. For every held-out target image we run:
    no patch      the answer in each of the four languages (the right concept in the right language)
    LogitLens     the summed output of a head set at the last prompt token, read against the
                  vocabulary: in which language its top token names the concept. Two head sets: the
                  32 gaze heads, and the top 100 published Gaze Heads. Also the cosine of the gaze
                  heads' summed output for one image in two languages, and for two images in English.
    cross         a source asks in another language about another image (27 cases per target, as in
                  language/main_experiments/03). Five head sets have their outputs copied from the
                  source: the gaze heads at every written token, the gaze heads at the first written
                  token only, the top 32 of the concept search of this task (08_concept_search.py),
                  32 random heads, and the 16 language heads of this task.
    same          a source in the target's own language, the gaze heads at every written token.
A cross case counts as "concept" when the target names the source's concept in its own language,
and as "language" when it names its own concept in the source's language.

    python gaze/appendix_experiments/09_what_the_gaze_heads_bring_back.py --num_pairs 500
    python gaze/appendix_experiments/09_what_the_gaze_heads_bring_back.py --task objects --num_pairs 100

--num_pairs is the number of target images (500 colored-shape images, 100 object images in the paper).

Paper, colored shapes: the top token of the published top 100 names the shape in English in 100.0%
of German and 84.2% of French questions (otherwise in Chinese). Gaze heads, every token: concept
82.0%, language 0.0%, unchanged 13.8%; first token only: concept 67.7%; the search's own top 32:
concept in every case; random heads: concept 0.0%, language 0.3%; language heads: language 95.3%,
concept 0.0%; a source in the same language: concept 86.8%.
Paper, objects: no patch 99.0%; cosines 0.96 and 0.36; the top token of the gaze heads names the
object in English in 48.8% of readouts, starts its English name in 33.0%, names it in Chinese in
2.0%; gaze heads: concept 9.8%, and in 15.5% the target names the object its own image holds at the
source's asked panel; the search's top 32: concept 86.1%, language 0.0%; random heads 0.0%;
language heads: language 87.6%, concept 0.0%.
The 67.7%, the search's top 32 on colored shapes, and all object numbers come from an earlier run
(100 colored-shape images) with an earlier list of gaze heads (HEADS["gaze_coco"], 31 of 32 shared),
an earlier concept search, and, for objects, HEADS["language_objects"]. This file uses HEADS["gaze"].
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import LANGUAGES, task_argument
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import all_heads, load_model
from utils.patching import top_tokens, write_tokens, written_by_heads


def top_word(model, task, run, heads, concept):
    """LogitLens of the heads' summed output: the language its top token names the concept in, or
    "start" when the token is the start of the English name."""
    token = top_tokens(model, written_by_heads(model, run, heads, "last"), 1)[0]
    language = task.answer_language(token, concept)
    piece = token.strip(" \t-_").lower()
    if language == "other" and piece and any(word.startswith(piece) for word in task.words["en"][concept]):
        return "start"
    return language


def main():
    task = task_argument()
    args = parse_args(__doc__, num_pairs=500 if task.name == "colors" else 100)
    model = load_model(args.model, args.device)
    gaze, every_head = HEADS["gaze"], all_heads(model)
    search = HEADS["concept_search" if task.name == "colors" else "concept_search_objects"][:32]
    language_heads = HEADS["language" if task.name == "colors" else "language_objects"]
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    results = Results(f"what_the_gaze_heads_bring_back_{task.name}", args)
    for spec in task.images(args.split, args.start + args.num_pairs)[args.start:]:
        concept = task.concept(spec)
        runs = {language: task.run(model, spec, language) for language in LANGUAGES}
        for language, run in runs.items():
            results.add(spec["index"], kind="no patch", language=language,
                        no_patch_right=task.answer_language(write_tokens(model, run), concept) == language,
                        lens_gaze=top_word(model, task, run, gaze, concept),
                        lens_published=top_word(model, task, run, HEADS["published_gaze"][:100], concept))
        sums = {language: written_by_heads(model, run, gaze, "last") for language, run in runs.items()}
        results.add(spec["index"], kind="cosine", across_languages=float(np.mean(
            [cos(sums[a], sums[b]) for i, a in enumerate(LANGUAGES) for b in LANGUAGES[i + 1:]])),
            across_images=float(np.mean([cos(sums["en"], written_by_heads(model, task.run(model, s, "en"), gaze, "last"))
                                         for s in spec["sources"]])))
        del runs                                         # free the GPU memory before the cases
        random_heads = [every_head[i] for i in np.random.default_rng(11000 + spec["index"]).permutation(len(every_head))[:32]]
        arms = {"gaze": (gaze, True), "gaze_first_token": (gaze, False), "search_top32": (search, True),
                "random": (random_heads, True), "language_heads": (language_heads, True)}
        for source, target, case in task.cases(model, spec):
            row = {"kind": "cross", **case}
            for arm, (heads, every_token) in arms.items():
                text = write_tokens(model, target, source, heads, every_token=every_token)
                outcome = task.patched_answer(text, case)
                row.update({f"{arm}_concept": outcome == "concept", f"{arm}_language": outcome == "language",
                            f"{arm}_unchanged": outcome == "unchanged", f"{arm}_text": text})
                if arm == "gaze":                       # an answer that names the target's item at the source's panel
                    row["gaze_item_at_source_panel"] = outcome == "other" and task.names_concept(
                        text, task.item_at(spec, case["source_panel"]))
            results.add(spec["index"], **row)
        for source, target, case in task.cases(model, spec, target_languages=LANGUAGES, same_language=True):
            text = write_tokens(model, target, source, gaze)
            results.add(spec["index"], kind="same", same_language_concept=task.same_language_answer(text, case) == "concept",
                        text=text, **case)
    colors = task.name == "colors"
    results.report("no_patch_right", paper=100.0 if colors else 99.0, label="no patch: right concept, right language")
    for language in LANGUAGES:
        for heads in ("gaze", "published"):
            rows = [r for r in results.rows if r["kind"] == "no patch" and r["language"] == language]
            counts = {k: sum(r[f"lens_{heads}"] == k for r in rows) for k in ("en", "start", "zh", "de", "fr", "other")}
            print(f"LogitLens, {heads:9s} heads, question in {language}: top token " +
                  ", ".join(f"{k} {100 * v / max(1, len(rows)):.1f}%" for k, v in counts.items()))
    print("paper: published top 100, colored shapes: English in 100.0% (de) and 84.2% (fr); gaze heads, objects: "
          "English 48.8%, start 33.0%, Chinese 2.0% over all languages")
    cosines = [r for r in results.rows if r["kind"] == "cosine"]
    for name, paper in (("across_languages", "0.95" if colors else "0.96"), ("across_images", "0.50" if colors else "0.36")):
        print(f"cosine {name}: {np.mean([r[name] for r in cosines]):.3f}   paper: {paper}")
    paper = ({"gaze_concept": 82.0, "gaze_language": 0.0, "gaze_unchanged": 13.8, "gaze_first_token_concept": 67.7,
              "search_top32_concept": 100.0, "random_concept": 0.0, "random_language": 0.3, "language_heads_language": 95.3,
              "language_heads_concept": 0.0, "same_language_concept": 86.8} if colors else
             {"gaze_concept": 9.8, "gaze_item_at_source_panel": 15.5, "search_top32_concept": 86.1,
              "search_top32_language": 0.0, "random_concept": 0.0, "language_heads_language": 87.6, "language_heads_concept": 0.0})
    for condition in ("gaze_concept", "gaze_language", "gaze_unchanged", "gaze_item_at_source_panel", "gaze_first_token_concept",
                      "search_top32_concept", "search_top32_language", "random_concept", "random_language",
                      "language_heads_language", "language_heads_concept", "same_language_concept"):
        results.report(condition, paper=paper.get(condition))
    results.save()


if __name__ == "__main__":
    main()
