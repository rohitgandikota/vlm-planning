"""What keeps Chinese runs Chinese when the language heads are zeroed (Appendix F.8, Table 17).

Zeroing the 16 language heads turns German and French answers English, but not Chinese ones. To
find what else carries the language, we take a run asked in German, French, or Chinese with the 16
heads zeroed at the last prompt token, and copy one more component from a run on the same image
asked in English, so that the two runs differ only in language. We score the first answer token by
the gain of logit(English word) - logit(word in the question's language) over the zeroed run.
    1. Search, on the 12 search images: the residual entering each layer, each MLP output, all other
       heads of a layer together, and every other head alone, each copied from the English run.
       The other heads are ranked by their mean gain, separately for each language.
    2. Test, on the held-out images, while the target writes: the 16 heads zeroed, together with the
       top 16 or 64 heads of step 1 (or 64 random heads), at every written token; the 64 heads alone;
       the residual entering layer 2, 4, 8, or 16 at the last prompt token set to its average over
       the search images asked in English, with the 16 heads zeroed or alone.
A run of step 2 counts when its answer names its concept in English.

    python language/appendix_experiments/07_what_keeps_chinese_runs_chinese.py --num_pairs 500

Paper (percent English, German / French / Chinese): 16 heads zeroed 99.6 / 88.4 / 0; and 16 more
heads 100 / 100 / 0; and 64 more heads 100 / 100 / 90; and 64 random heads 92 / 85 / 0; the 64 more
heads alone 0 / 12 / 0; 16 zeroed and the residual at layer 4 set 100 / 100 / 31, at layer 8
100 / 100 / 89, at layer 16 100 / 100 / 100; the residual at layer 8 alone 0 / 0 / 0. In step 1, the
residual closes most of the gap from layer 4 on, and MLPs 22-35 each move the answer.
These numbers come from an earlier run on 100 images with an earlier list of language heads
(HEADS["language_coco"]). This file uses HEADS["language"]. The first row of the paper's table is
zeroing at the first token only (from 05_zeroing_and_setting_heads.py); here the heads are zeroed
at every written token.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import first_answer_tokens, language_task
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import all_heads, load_model, n_heads, n_layers
from utils.patching import Patch, run_patches, set_head_outputs, write_with_patch

LANGUAGES = ("de", "fr", "zh")
EXTRA, DEPTHS = (16, 64), (2, 4, 8, 16)
PAPER = {"zeroed": "99.6 / 88.4 / 0", "zeroed_and_16_more": "100 / 100 / 0", "zeroed_and_64_more": "100 / 100 / 90",
         "zeroed_and_64_random": "92 / 85 / 0", "64_more_alone": "0 / 12 / 0", "zeroed_and_residual_4": "100 / 100 / 31",
         "zeroed_and_residual_8": "100 / 100 / 89", "zeroed_and_residual_16": "100 / 100 / 100", "residual_8_alone": "0 / 0 / 0",
         "zeroed_and_16_random": None, "16_more_alone": None, "zeroed_and_residual_2": None, "residual_2_alone": None,
         "residual_4_alone": None, "residual_16_alone": None}


def search(model, task, core, zero, tokens, language):
    """Step 1 for one language: the mean gain of every copied component. Returns (gains, other heads ranked)."""
    layers, heads = n_layers(model), n_heads(model)
    gains = {"residual": torch.zeros(layers + 1), "mlp": torch.zeros(layers), "layer_heads": torch.zeros(layers),
             "head": torch.zeros(layers, heads)}
    images = task.images("search")
    for spec in images:
        target, english = task.run(model, spec, language), task.run(model, spec, "en")
        last, concept = target.last, task.concept(spec)

        def zeroed():                                        # a patch with the 16 heads zeroed
            patch = Patch()
            set_head_outputs(core, zero)(patch, last)
            return patch

        def copy_head(patch, l, h):                          # one head from the English run (the 16 stay zero)
            return patch if (l, h) in core else patch.head_output(l, h, last, english.head_output(l, h, english.last))

        patches = [zeroed()]
        patches += [zeroed().residual(b, last, english.residual(b, english.last)) for b in range(layers + 1)]
        patches += [zeroed().mlp(l, last, english.mlp_output(l, english.last)) for l in range(layers)]
        for l in range(layers):                              # all heads of one layer
            patch = zeroed()
            for h in range(heads):
                copy_head(patch, l, h)
            patches.append(patch)
        patches += [copy_head(zeroed(), l, h) for l in range(layers) for h in range(heads)]
        logits = run_patches(model, target, patches, max_batch=32)["logits"]
        margin = logits[:, tokens["en"][concept]] - logits[:, tokens[language][concept]]
        gain = margin[1:] - margin[0]
        gains["residual"] += gain[:layers + 1]
        gains["mlp"] += gain[layers + 1:2 * layers + 1]
        gains["layer_heads"] += gain[2 * layers + 1:3 * layers + 1]
        gains["head"] += gain[3 * layers + 1:].view(layers, heads)
    gains = {name: value / len(images) for name, value in gains.items()}
    order = torch.argsort(gains["head"].flatten(), descending=True).tolist()
    ranked = [(i // heads, i % heads) for i in order if (i // heads, i % heads) not in core]
    return gains, ranked


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    task, core = language_task("colors"), HEADS["language"]
    tokens = first_answer_tokens(model, task)
    zero = dict.fromkeys(all_heads(model), torch.zeros(model.config.text_config.head_dim, device=model.device))
    ranked, english_residual = {}, 0
    for language in LANGUAGES:
        gains, ranked[language] = search(model, task, core, zero, tokens, language)
        print(f"{language}: gain of the residual entering each layer {[round(float(g), 1) for g in gains['residual']]}")
        print(f"{language}: gain of each MLP {[round(float(g), 1) for g in gains['mlp']]}")
        print(f"{language}: top 16 other heads {ranked[language][:16]}")
    for spec in task.images("search"):                     # the residual of the English runs, entering every layer
        run = task.run(model, spec, "en")
        english_residual = english_residual + torch.stack([run.residual(b, run.last) for b in range(n_layers(model) + 1)])
    english_residual = english_residual / len(task.images("search"))
    others = [head for head in all_heads(model) if head not in core]
    results = Results("what_keeps_chinese_runs_chinese", args)
    for spec in task.images(args.split, args.start + args.num_pairs)[args.start:]:
        rng = np.random.default_rng(9000 + spec["index"])
        for language in LANGUAGES:
            run, concept = task.run(model, spec, language), task.concept(spec)
            pool = [others[i] for i in rng.permutation(len(others))]
            arms = {"zeroed": core, **{f"zeroed_and_{k}_more": core + ranked[language][:k] for k in EXTRA},
                    **{f"zeroed_and_{k}_random": core + pool[:k] for k in EXTRA},
                    **{f"{k}_more_alone": ranked[language][:k] for k in EXTRA}}
            row = {}
            for name, heads in arms.items():
                text = write_with_patch(model, run, set_head_outputs(heads, zero), every_token=True)
                row[name] = task.answer_language(text, concept) == "en"
            for depth in DEPTHS:
                for name, heads in ((f"zeroed_and_residual_{depth}", core), (f"residual_{depth}_alone", [])):
                    def edit(patch, token, heads=heads, depth=depth):
                        set_head_outputs(heads, zero)(patch, token)
                        if token == run.last:                    # the residual is set at the last prompt token only
                            patch.residual(depth, token, english_residual[depth])
                    row[name] = task.answer_language(write_with_patch(model, run, edit, every_token=True), concept) == "en"
            results.add(spec["index"], language=language, **row)
    for name, paper in PAPER.items():
        rates = [100 * np.mean([r[name] for r in results.rows if r["language"] == language]) for language in LANGUAGES]
        print(f"{name:26s} English: " + " / ".join(f"{rate:5.1f}" for rate in rates) + f"   paper: {paper}")
    results.save(ranked_other_heads={language: heads[:256] for language, heads in ranked.items()})


if __name__ == "__main__":
    main()
