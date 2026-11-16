"""Choosing the language with no source (Appendix F.8, Figure 12).

No source run is used. In a run asked in one language, we set the outputs of the top K heads of the
language-head search to their average output under another language, at the last prompt token and
every written token (K = 8, 16, 32, 64). The averages are taken at the last prompt token over the 12
search images. A run counts when its answer names its own concept in that other language. Every
pair of languages is run, except from German, French, or Chinese into English, which
05_zeroing_and_setting_heads.py covers. As a control, 16 random heads get the same averages.

    python language/appendix_experiments/06_setting_heads_to_another_language.py --num_pairs 500
    python language/appendix_experiments/06_setting_heads_to_another_language.py --task objects --num_pairs 100

Paper: with 16 heads, the answer turns to the other language in 77.3% of runs on colored shapes and
85.2% on objects; random heads with the same averages 0.0%. Chinese runs, which stay Chinese when the
heads are zeroed, switch to English, German, or French. Leaving English is the hardest direction and
needs 32 heads instead of 16. These numbers come from an earlier run on 100 images with an earlier
ranking of the language heads. The random heads are drawn here from all heads; the earlier run drew
them from that ranking.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import LANGUAGES, average_head_outputs, task_argument
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import all_heads, load_model
from utils.patching import set_head_outputs, write_with_patch

SIZES = (8, 16, 32, 64)


def main():
    task = task_argument()
    args = parse_args(__doc__, num_pairs=500 if task.name == "colors" else 100)
    model = load_model(args.model, args.device)
    ranking = HEADS["language_ranking" if task.name == "colors" else "language_objects_ranking"]
    every_head = all_heads(model)
    averages = average_head_outputs(model, task, every_head)             # every head, so the random heads have theirs too
    results = Results(f"setting_heads_to_another_language_{task.name}", args)
    for spec in task.images(args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = [every_head[i] for i in np.random.default_rng(7000 + spec["index"]).permutation(len(every_head))[:16]]
        concept = task.concept(spec)
        for language in LANGUAGES:
            run = task.run(model, spec, language)
            for other in LANGUAGES:
                if other == language or (language != "en" and other == "en"):
                    continue
                row = {}
                for k in SIZES:
                    text = write_with_patch(model, run, set_head_outputs(ranking[:k], averages[other]), every_token=True)
                    row[f"top{k}"] = task.answer_language(text, concept) == other
                text = write_with_patch(model, run, set_head_outputs(random_heads, averages[other]), every_token=True)
                row["random16"] = task.answer_language(text, concept) == other
                results.add(spec["index"], language=language, other=other, **row)
    for k in SIZES:
        results.report(f"top{k}", paper={16: 77.3 if task.name == "colors" else 85.2}.get(k), label=f"top {k} heads set: other language")
    results.report("random16", paper=0.0, label="16 random heads set: other language")
    for language in LANGUAGES:
        for other in LANGUAGES:
            rows = [r for r in results.rows if r["language"] == language and r["other"] == other]
            if rows:
                print(f"{language} -> {other}: " + ", ".join(f"top {k} {100 * np.mean([r[f'top{k}'] for r in rows]):5.1f}%" for k in SIZES))
    results.save()


if __name__ == "__main__":
    main()
