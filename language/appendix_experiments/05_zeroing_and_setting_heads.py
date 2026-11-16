"""Zeroing the language heads, against setting them to their English average (Appendix F.8, Figure 11).

No source run is used. In runs asked in German, French, or Chinese, we change the outputs of the
top K heads of the language-head search (HEADS["language_ranking"]):
    zeroed, first token    the top 16 heads set to zero at the last prompt token only
    zeroed                 the top K heads set to zero at the last prompt token and every written
                           token, K = 16, 32, 64, 128, 256
    random zeroed          K random heads set to zero in the same way
    set to English         the top K heads set to their average output under the English question,
                           at every token, K = 4, 8, 16, 32, 64
The averages are taken at the last prompt token over the 12 search images. A run counts when the
answer names its concept in English; "broken" when it names it in no language.

    python language/appendix_experiments/05_zeroing_and_setting_heads.py --num_pairs 500
    python language/appendix_experiments/05_zeroing_and_setting_heads.py --task objects --num_pairs 100

Paper, colored shapes: zeroing 16 heads at the first token returns English in 99.6% of German,
88.4% of French, and 0.0% of Chinese runs (62.7% over all three; objects 23.7%). Zeroing 32 or 64
heads brings French to 100% and leaves Chinese at 0%; zeroing more heads breaks the output;
zeroing random heads never returns English. Setting 16 heads to English returns English in every
run of both tasks. Only the first-token zeroing of colored shapes was run on the 500 images; the
rest comes from an earlier run on 100 images with an earlier ranking of the language heads. The
random heads are drawn here from all heads; the earlier run drew them from that ranking.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import average_head_outputs, task_argument
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import all_heads, load_model
from utils.patching import set_head_outputs, write_with_patch

ZERO_SIZES = (16, 32, 64, 128, 256)
MEAN_SIZES = (4, 8, 16, 32, 64)


def main():
    task = task_argument()
    args = parse_args(__doc__, num_pairs=500 if task.name == "colors" else 100)
    model = load_model(args.model, args.device)
    ranking = HEADS["language_ranking" if task.name == "colors" else "language_objects_ranking"]
    english = average_head_outputs(model, task, ranking[:max(MEAN_SIZES)], ("en",))["en"]
    every_head = all_heads(model)
    results = Results(f"zeroing_and_setting_heads_{task.name}", args)
    for spec in task.images(args.split, args.start + args.num_pairs)[args.start:]:
        order = np.random.default_rng(7000 + spec["index"]).permutation(len(every_head))
        concept = task.concept(spec)
        for language in ("de", "fr", "zh"):
            run = task.run(model, spec, language)
            zero = dict.fromkeys(every_head, torch.zeros_like(run.head_output(0, 0, "last")))
            arms = {"zeroed_16_first_token": (set_head_outputs(ranking[:16], zero), False)}
            for k in ZERO_SIZES:
                arms[f"zeroed_{k}"] = (set_head_outputs(ranking[:k], zero), True)
                arms[f"random_zeroed_{k}"] = (set_head_outputs([every_head[i] for i in order[:k]], zero), True)
            for k in MEAN_SIZES:
                arms[f"set_to_english_{k}"] = (set_head_outputs(ranking[:k], english), True)
            row = {}
            for name, (edit, every_token) in arms.items():
                answer = task.answer_language(write_with_patch(model, run, edit, every_token=every_token), concept)
                row[name], row[f"{name}_broken"] = answer == "en", answer == "other"
            results.add(spec["index"], language=language, **row)
    for language, paper in (("de", 99.6), ("fr", 88.4), ("zh", 0.0)):
        rows = [r for r in results.rows if r["language"] == language]
        print(f"zeroed 16, first token, {language}: English in {100 * np.mean([r['zeroed_16_first_token'] for r in rows]):.1f}%   paper: {paper}")
    results.report("zeroed_16_first_token", paper=62.7 if task.name == "colors" else 23.7, label="zeroed 16, first token: English")
    for k in ZERO_SIZES:
        results.report(f"zeroed_{k}", label=f"zeroed {k}: English")
        results.report(f"zeroed_{k}_broken", label=f"zeroed {k}: broken")
        results.report(f"random_zeroed_{k}", paper=0.0, label=f"random zeroed {k}: English")
    for k in MEAN_SIZES:
        results.report(f"set_to_english_{k}", paper=100.0 if k == 16 else None, label=f"set to English {k}: English")
    results.save()


if __name__ == "__main__":
    main()
