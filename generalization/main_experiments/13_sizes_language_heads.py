"""The language heads on the 2B and 4B (Appendix G.1.5, Table "sizes-language", rows 3 to 5).

The test of language/main_experiments/04_language_heads.py with each size's own 16 language heads.
A source asks in one language about one image, a target asks in another language about another
image. While the target writes, the outputs of the language heads come from the source, at the last
prompt token and at every written token. Outcomes (see `patched_answer`):
    language   the target's shape in the source's language: the language moved
    word       the source's shape in the source's language: the source's word was copied
A second patch holds the late MLPs: the language heads' outputs come from the source at the last
prompt token, and the outputs of the MLPs of the last three layers there are the target's own. If
these MLPs turn the language heads' signal into the word, the language no longer moves.

    python generalization/main_experiments/13_sizes_language_heads.py --model Qwen/Qwen3-VL-4B-Instruct --num_pairs 500

--num_pairs is the number of target images, 27 cases each (the paper's runs used 100 held-out
images). Models: Qwen3-VL-2B (16 GB) or 4B (24 GB). 2 written answers of up to 8 tokens per case.

Paper: the language moves in 23.7% (2B), 93.0% (4B), 95.3% (8B); copied source words 55.7%, 2.6%,
0.0%; with the late MLPs held 0.4%, 10.5%, 0.9%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import color_cases, color_images, patched_answer, run_color_question
from utils.experiment import Results, parse_args
from utils.heads import heads_of_size
from utils.model import load_model, n_layers
from utils.patching import make_patch, write_tokens, write_with_fixed_patch

PAPER = {"language_heads": {"2B": 23.7, "4B": 93.0, "8B": 95.3}, "word_copied": {"2B": 55.7, "4B": 2.6, "8B": 0.0},
         "late_mlps_held": {"2B": 0.4, "4B": 10.5, "8B": 0.9}}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    language_heads = heads_of_size(args.model, "language")
    late = range(n_layers(model) - 3, n_layers(model))
    size = args.model.split("-")[-2]
    results = Results(f"sizes_language_heads_{args.model.split('/')[-1]}", args)
    for spec in color_images(args.split, args.start + args.num_pairs)[args.start:]:
        recorded = {language: run_color_question(model, spec, language) for language in ("de", "fr", "zh")}
        for source, target, case in color_cases(model, spec):
            outcome = patched_answer(write_tokens(model, target, source, language_heads), case)
            held = make_patch(source, target, "head_output", "last", heads=language_heads)
            for layer in late:                                              # the target's own late MLP outputs
                held.add(make_patch(recorded[case["target_language"]], target, "mlp", "last", layer=layer))
            held_outcome = patched_answer(write_with_fixed_patch(model, target, held), case)
            results.add(spec["index"], language_heads=outcome == "language", word_copied=outcome == "word",
                        late_mlps_held=held_outcome == "language", outcome=outcome, outcome_held=held_outcome, **case)
    for name, label in (("language_heads", "language heads: target's shape, source's language"),
                        ("word_copied", "language heads: the source's word copied"),
                        ("late_mlps_held", "late MLPs held: target's shape, source's language")):
        results.report(name, paper=PAPER[name].get(size), label=label)
    results.save(language_heads=language_heads)


if __name__ == "__main__":
    main()
