"""On comic images, the same binding and relay steps appear (Appendix G.3.1).

The comic version of the one-answer task (see 20_comic_residual_by_layer.py), with the heads that
were selected on it (HEADS["comic_query"], ["comic_key"], ["comic_output"], ["comic_relay"]). On
the held-out comic pairs we patch, as in Sections 4.1 and 4.2:
    the queries of the 8 query heads at the shape token            P  (the source's request)
    the 8 key heads read the source's image keys at the shape token R  (the source's layout)
    the outputs of the 32 output heads at the shape token           K  (the source's panel)
    the relay heads' values at the colon token, from the run with the output-head patch      K
    the relay heads' values at the colon token, from the source run                          K
    the output-head patch with the target's own relay values restored                        T
    the output of the first MLP (layer 0) at the shape token        P
Each attention patch also has a control with the same number of random heads from the same layers.
For each patch we also read which panel the published Gaze Heads look at from the colon token.

    python generalization/main_experiments/21_comic_binding_and_relay.py --comics /path/to/comics

--comics: a folder with comic1/p1.png ... comic500/p4.png (see 20_comic_residual_by_layer.py).
The held-out comic pairs are pairs 150 to 249 (200 cases); every pair is used.

Paper: queries 97.5%, keys 90.0%, outputs 85.5%; the published Gaze Heads follow in 99.0%, 88.0%
and 84.5%. Relay values from the run with the output-head patch 85.5%, the source's own values
91.5%; restoring the target's values brings back its answer in 100.0%. The first MLP moves the
request in 99.0%. The strongest random control reaches 0.0%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import comic_both_directions, comic_pairs, folder_argument
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import make_patch, panel_attention, record_patch, run_patches


def relay_values(run, target, heads):
    """At the colon token, `heads` read the values of `run` from the shape token through the token before the colon token."""
    return make_patch(run, target, "values", "colon", heads=heads, positions=range(target.token("shape"), target.token("colon")))


def attention_patches(model, source, target, heads):
    """The query, key, and output patches at the shape token, and the two relay patches, with the outcome each should give."""
    outputs = make_patch(source, target, "head_output", "shape", heads=heads["output"])
    patched = record_patch(model, target, outputs)                        # the run with the output-head patch
    return {"queries": (make_patch(source, target, "query", "shape", heads=heads["query"]), "P"),
            "keys": (make_patch(source, target, "image_keys", "shape", heads=heads["key"]), "R"),
            "outputs": (outputs, "K"),
            "relay_from_patched_run": (relay_values(patched, target, heads["relay"]), "K"),
            "relay_from_source": (relay_values(source, target, heads["relay"]), "K")}


def main():
    comics = folder_argument("--comics", "a folder with comic1/p1.png ... comic500/p4.png")
    args = parse_args(__doc__, num_pairs=100)
    model = load_model(args.model, args.device)
    results = Results("comic_binding_and_relay", args)
    comic_heads = {name: HEADS["comic_" + name] for name in ("query", "key", "output", "relay")}
    for pair in comic_pairs(args.split, args.start + args.num_pairs)[args.start:]:
        random_heads = {name: random_like(heads, seed=pair["index"]) for name, heads in comic_heads.items()}
        for source, target, letters in comic_both_directions(model, comics, pair):
            found = attention_patches(model, source, target, comic_heads)
            control = attention_patches(model, source, target, random_heads)
            restored = make_patch(source, target, "head_output", "shape", heads=comic_heads["output"])
            restored.add(relay_values(target, target, comic_heads["relay"]))
            first_mlp = make_patch(source, target, "mlp", "shape", layer=0)
            patches = [patch for patch, _ in found.values()] + [restored, first_mlp] + [patch for patch, _ in control.values()]
            result = run_patches(model, target, patches, keep_attention=True)
            answers = result["answers"]
            looked_at = panel_attention(result["attention"], HEADS["published_gaze"], target).argmax(dim=1).tolist()
            row = {}
            for i, (name, (_, outcome)) in enumerate(found.items()):
                row[name] = answers[i] == " " + letters[outcome]
                row[name + "_gaze_follows"] = looked_at[i] == target.image.letters.index(letters[outcome])
            row["target_values_restored"] = answers[len(found)] == " " + letters["T"]
            row["first_mlp"] = answers[len(found) + 1] == " " + letters["P"]
            row["random_heads_any"] = any(answer == " " + letters[outcome] for answer, (_, outcome)
                                          in zip(answers[len(found) + 2:], control.values()))
            results.add(pair["index"], **row)
    for name, paper in (("queries", 97.5), ("keys", 90.0), ("outputs", 85.5), ("queries_gaze_follows", 99.0),
                        ("keys_gaze_follows", 88.0), ("outputs_gaze_follows", 84.5), ("relay_from_patched_run", 85.5),
                        ("relay_from_source", 91.5), ("target_values_restored", 100.0), ("first_mlp", 99.0), ("random_heads_any", 0.0)):
        results.report(name, paper=paper)
    results.save()


if __name__ == "__main__":
    main()
