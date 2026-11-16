"""The heads selected on comic images still work on the paper's blank panels (Appendix G.3.1).

The heads of the comic version of the task (21_comic_binding_and_relay.py) are used unchanged on the
paper's one-answer pairs, the row of blank panels with the paper's prompt:
    the 8 comic query heads' queries at the shape token                  P  (the source's request)
    the 8 comic key heads read the source's image keys at the shape token R  (the source's layout)
    the 32 comic output heads' outputs at the shape token                 K  (the source's panel)
    the 16 comic relay heads' values at the colon token, from the run with that output patch   K

    python generalization/main_experiments/22_comic_heads_on_panels.py --num_pairs 500

Paper: the queries move the request in 67.0% of cases, the keys the layout in 81.0%, the outputs
the pointer in 85.0%, and the relay values in 87.5%. These numbers come from an earlier, smaller run
on 100 held-out pairs of an earlier pair set, so expect values near them, not the same values.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import make_patch, record_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("comic_heads_on_panels", args)
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, letters in both_directions(model, pair):
            output_patch = make_patch(source, target, "head_output", "shape", heads=HEADS["comic_output"])
            patched = record_patch(model, target, output_patch)
            read_from = range(target.token("shape"), target.token("colon"))
            patches = [make_patch(source, target, "query", "shape", heads=HEADS["comic_query"]),
                       make_patch(source, target, "image_keys", "shape", heads=HEADS["comic_key"]),
                       output_patch,
                       make_patch(patched, target, "values", "colon", heads=HEADS["comic_relay"], positions=read_from)]
            queries, keys, outputs, relay = run_patches(model, target, patches)["answers"]
            results.add(pair["index"], comic_queries=queries == " " + letters["P"], comic_keys=keys == " " + letters["R"],
                        comic_outputs=outputs == " " + letters["K"], comic_relay_values=relay == " " + letters["K"])
    results.report("comic_queries", paper=67.0, label="comic query heads: request moved (P)")
    results.report("comic_keys", paper=81.0, label="comic key heads: layout moved (R)")
    results.report("comic_outputs", paper=85.0, label="comic output heads: pointer moved (K)")
    results.report("comic_relay_values", paper=87.5, label="comic relay heads: pointer moved (K)")
    results.save()


if __name__ == "__main__":
    main()
