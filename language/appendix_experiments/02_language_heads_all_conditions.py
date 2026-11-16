"""Language heads: every condition, on both tasks (Appendix F.6, Table 16, first ten rows).

This extends language/main_experiments/01 and 04. A source asks in language A about image X, and a
target asks in language B about image Y (B is German, French, or Chinese; 27 cases per target
image). The outputs of a head set at the last prompt token are copied from the source while the
target writes its answer. A case counts as "language" when the target names its own concept in the
source's language, and as "word" when it copies the source's word. The conditions:
    every token        the 16 language heads, at the last prompt token and every written token
    first token        the same heads, at the last prompt token only
    random             16 random heads of any layer, one fixed draw (first token)
    lowest             the 16 lowest-ranked heads of the search (first token)
    MLPs 33-35 held    the language heads patched, and the outputs of MLPs 33, 34, 35 at the last
                       prompt token kept at the target's own values (first token)
    MLPs 28-32 held    the same with MLPs 28 to 32
    attention held     the same with all other heads of layers 33 to 35 kept at the target's values
    same language      a source in the target's own language (first token): counts when the word is unchanged
and the answers with no patch, in the language of the question.

    python language/appendix_experiments/02_language_heads_all_conditions.py --num_pairs 500
    python language/appendix_experiments/02_language_heads_all_conditions.py --task objects --num_pairs 100

Paper, colored shapes / objects: no patch 100.0 / 99.0; every token 95.3 / 87.6; first token 90.7 /
78.4; copied words 0.0 / 1.3; random 0.0 / 0.5; lowest 0.0 / 0.3; same language unchanged 100.0 /
91.1; MLPs 33-35 held 0.9 / 5.0; MLPs 28-32 held 10.7 / 4.4; attention held 91.3 / 79.7.
On colored shapes, only the no-patch, every-token, random, and MLPs 33-35 rows were run on the 500
images; the others, and the whole objects column, come from an earlier run on 100 images with an
earlier list of language heads (HEADS["language_coco"] for colored shapes, HEADS["language_objects"]
for objects). This file uses HEADS["language"] on colored shapes.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import LANGUAGES, task_argument
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import all_heads, load_model, n_heads
from utils.patching import write_tokens, write_with_patch

PAPER = {"colors": {"no_patch": 100.0, "every_token": 95.3, "first_token": 90.7, "word_copied": 0.0, "random": 0.0,
                    "lowest": 0.0, "same_language_unchanged": 100.0, "late_mlps_held": 0.9, "mid_mlps_held": 10.7,
                    "late_attention_held": 91.3},
         "objects": {"no_patch": 99.0, "every_token": 87.6, "first_token": 78.4, "word_copied": 1.3, "random": 0.5,
                     "lowest": 0.3, "same_language_unchanged": 91.1, "late_mlps_held": 5.0, "mid_mlps_held": 4.4,
                     "late_attention_held": 79.7}}


def heads_and_held(model, source, target, heads, mlps=(), attention_layers=()):
    """An edit for `write_with_patch`: the heads take the source's outputs at the last prompt token,
    and these MLPs and all other heads of these layers keep the target's own outputs."""
    def edit(patch, token):
        for layer in mlps:
            patch.mlp(layer, token, target.mlp_output(layer, token))
        for layer in attention_layers:
            for head in range(n_heads(model)):
                patch.head_output(layer, head, token, target.head_output(layer, head, token))
        for layer, head in heads:                     # added last, so it wins over the held heads
            patch.head_output(layer, head, token, source.head_output(layer, head, source.last))
    return edit


def main():
    task = task_argument()
    args = parse_args(__doc__, num_pairs=500 if task.name == "colors" else 100)
    model = load_model(args.model, args.device)
    every_head = all_heads(model)
    language = HEADS["language" if task.name == "colors" else "language_objects"]
    lowest = HEADS["language_lowest" if task.name == "colors" else "language_objects_lowest"]
    random_heads = [every_head[i] for i in np.random.default_rng(1016).choice(len(every_head), 16, replace=False)]
    results = Results(f"language_heads_all_conditions_{task.name}", args)
    for spec in task.images(args.split, args.start + args.num_pairs)[args.start:]:
        for code in LANGUAGES:
            text = write_tokens(model, task.run(model, spec, code, record=False))
            results.add(spec["index"], kind="no patch", no_patch=task.answer_language(text, task.concept(spec)) == code)
        for source, target, case in task.cases(model, spec):
            texts = {"every_token": write_tokens(model, target, source, language),
                     "first_token": write_tokens(model, target, source, language, every_token=False),
                     "random": write_tokens(model, target, source, random_heads, every_token=False),
                     "lowest": write_tokens(model, target, source, lowest, every_token=False),
                     "late_mlps_held": write_with_patch(model, target, heads_and_held(model, source, target, language, mlps=(33, 34, 35))),
                     "mid_mlps_held": write_with_patch(model, target, heads_and_held(model, source, target, language, mlps=range(28, 33))),
                     "late_attention_held": write_with_patch(model, target, heads_and_held(
                         model, source, target, language, attention_layers=(33, 34, 35)))}
            outcomes = {name: task.patched_answer(text, case) for name, text in texts.items()}
            results.add(spec["index"], kind="cross", **case, **{name: o == "language" for name, o in outcomes.items()},
                        word_copied=outcomes["every_token"] == "word", outcomes=outcomes)
        for source, target, case in task.cases(model, spec, same_language=True):
            text = write_tokens(model, target, source, language, every_token=False)
            results.add(spec["index"], kind="same", same_language_unchanged=task.same_language_answer(text, case) == "unchanged", **case)
    for condition, paper in PAPER[task.name].items():
        results.report(condition, paper=paper)
    results.save()


if __name__ == "__main__":
    main()
