"""The rebind at every model size (Appendix G.1.4, Table "sizes-progress", rebind rows).

The design of progress/main_experiments/04 and 05: the target has its shape at panels 1, 3, 4 and 6
and has written the letters of panels 1 and 3. The source is the same image with two letters
swapped. At an answer token, a set of heads reads the source's image keys:
    last answer token    source: the letters of panels 3 and 4 swapped. If the cursor moves to
                         panel 4, the target answers with the letter of panel 6.
    first answer token   source: the letters of panels 1 and 4 swapped. The control: the cursor
                         should not move.
The head sets: the size's binding heads (on the 2B, 4B and 32B the paper's runs used the binding,
query and key heads together), every head of the model, and random heads from the same layers (one
draw, seed 5). A case counts as moved when the target answers with the letter of panel 6.

    python generalization/main_experiments/09_sizes_rebind.py --model Qwen/Qwen3-VL-4B-Instruct --num_pairs 500

The pairs are drawn for this experiment and kept by the model's own no-patch check
(`progress_pairs_of`), so --split is not used. The paper's runs used 200 pairs on the 2B and 4B and
100 on the 32B.
Models: any Qwen3-VL size. 2B 16 GB, 4B 24 GB, 8B 48 GB, 32B 160 GB (one B200) or --device cpu.
5 patched runs per pair.

Paper (2B / 4B / 8B / 32B): last answer token, binding heads 10.5 / 23.5 / 84.2 / 23.0; all heads
13.5 / 23.5 / 62.2 / 39.0; random heads 0.0 at every size; first answer token, binding heads / all
heads 0.0 / -- (2B), 0.0 / 7.0 (4B), 1.2 / 2.5 (8B), 0.0 / 5.0 (32B).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import name_answer, progress_pairs_of, run_list
from utils.experiment import Results, parse_args
from utils.heads import heads_of_size, random_heads_like
from utils.model import all_heads, load_model
from utils.patching import make_patch, run_patches

PAPER = {"last_binding": {"2B": 10.5, "4B": 23.5, "8B": 84.2, "32B": 23.0}, "last_all": {"2B": 13.5, "4B": 23.5, "8B": 62.2, "32B": 39.0},
         "first_binding": {"2B": 0.0, "4B": 0.0, "8B": 1.2, "32B": 0.0}, "first_all": {"4B": 7.0, "8B": 2.5, "32B": 5.0}}


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    binding = heads_of_size(args.model, "rebind")
    head_sets = {"binding": binding, "all": all_heads(model), "random": random_heads_like(model, binding, seed=5)}
    cells = [("last", name) for name in head_sets] + [("first", "binding"), ("first", "all")]
    results = Results(f"sizes_rebind_{args.model.split('/')[-1]}", args)
    for pair in progress_pairs_of(model, args.model, "cursor", args.start + args.num_pairs)[args.start:]:
        target = run_list(model, pair["target"])
        sources = {"last": run_list(model, pair["source"]), "first": run_list(model, pair["source_first"])}
        tokens = {"last": target.token("last_answer"), "first": target.token("answers")[0]}
        patches = [make_patch(sources[token], target, "image_keys", tokens[token], heads=head_sets[name]) for token, name in cells]
        answers = run_patches(model, target, patches, max_batch=4)["answers"]
        outcomes = {f"{token}_{name}": name_answer(answer, pair["letters"]) for (token, name), answer in zip(cells, answers)}
        results.add(pair["index"], outcomes=outcomes, **{cell: outcome == "after_panel_4" for cell, outcome in outcomes.items()})
    size = args.model.split("-")[-2]
    for token, name in cells:
        cell = f"{token}_{name}"
        results.report(cell, paper=PAPER.get(cell, {}).get(size, 0.0 if name == "random" else None),
                       label=f"{name} heads at the {token} answer token: letter of panel 6")
    results.save(binding_heads=binding)


if __name__ == "__main__":
    main()
