"""A longer question spreads binding over more tokens (Appendix G.3.4, Table 14).

In the paper's prompt only the question mark follows the shape token, and the binding patches act
on the shape token alone. With a longer question the binding heads may also act on the tokens that
follow it. For three wordings (the paper's, one sentence after the shape token, two sentences after
it) we patch the heads of the paper at
    S     the shape token
    A     the tokens of the question after the shape token
    S+A   both
with the outputs of the 16 binding heads, and with the queries and image keys of the 10 query and
key heads (outcome K, the source's panel). Then the relay heads at the colon token read their values,
from the shape token through the token before the colon token, out of the run with the binding
outputs patched at S or at S+A (K), and the binding patch at S+A with the target's own relay values
restored (T). Last, the relay values from the S+A run are installed only at S, or only at A (K).
Random heads from the same layers are patched at S+A as the control.

    python generalization/main_experiments/32_longer_question_spans.py

Pairs: the paper's pair generator asked in each wording, kept if both questions are answered right
in that wording. The paper's numbers come from an earlier, smaller run on 50 pairs (100 cases) per
wording that kept every pair, so expect values near them, not the same values. Default: 50 pairs.

Paper (paper's prompt / one sentence after / two sentences after): binding outputs at S 84.0 / 79.0
/ 52.0, at A 4.0 / 8.0 / 26.0, at S+A 87.0 / 88.0 / 87.0, random 1.0 / 0.0 / 0.0; queries and keys
at S 93.0 / 88.0 / 55.0, at A 2.0 / 2.0 / 11.0, at S+A 97.0 / 98.0 / 91.0; relay values from the S
run 86.0 / 80.0 / 58.0, from the S+A run 86.0 / 88.0 / 88.0, random 0.0, target's values restored
99.0 / 99.0 / 97.0. Relay values installed only at A: 67.0% with two sentences after; only at S:
0.0% with two sentences after, 47.3% with the paper's prompt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import make_variant, variant_both_directions, variant_one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS, random_like
from utils.model import load_model
from utils.patching import Patch, make_patch, record_patch, run_patches

WORDINGS = ("paper", "one_sentence_after", "two_sentences_after")
PAPER = {  # name: (paper's prompt, one sentence after, two sentences after); None where not reported
    "outputs_S": (84.0, 79.0, 52.0), "outputs_A": (4.0, 8.0, 26.0), "outputs_SA": (87.0, 88.0, 87.0),
    "outputs_SA_random": (1.0, 0.0, 0.0), "queries_and_keys_S": (93.0, 88.0, 55.0), "queries_and_keys_A": (2.0, 2.0, 11.0),
    "queries_and_keys_SA": (97.0, 98.0, 91.0), "queries_and_keys_SA_random": (1.0, 0.0, 0.0),
    "relay_from_S_run": (86.0, 80.0, 58.0), "relay_from_SA_run": (86.0, 88.0, 88.0), "relay_from_SA_run_random": (0.0, 0.0, 0.0),
    "relay_restored": (99.0, 99.0, 97.0), "relay_only_at_S": (47.3, None, 0.0), "relay_only_at_A": (None, None, 67.0)}


def at_tokens(source, target, component, heads, tokens):
    """One component of `heads` copied from the source at each of `tokens` (the same positions in both runs)."""
    patch = Patch()
    for token in tokens:
        if component == "queries_and_keys":
            patch.add(make_patch(source, target, "query", token, heads=heads))
            patch.add(make_patch(source, target, "image_keys", token, heads=heads))
        else:
            patch.add(make_patch(source, target, "head_output", token, heads=heads))
    return patch


def relay_values(run, target, heads, positions):
    return make_patch(run, target, "values", "colon", heads=heads, positions=positions)


def main():
    args = parse_args(__doc__, num_pairs=50)
    model = load_model(args.model, args.device)
    results = Results("longer_question_spans", args)
    for wording in WORDINGS:
        variant = make_variant("row", wording)
        for pair in variant_one_answer_pairs(model, variant, args.split, args.start + args.num_pairs)[args.start:]:
            for source, target, letters in variant_both_directions(model, pair, variant):
                shape, colon = target.token("shape"), target.token("colon")
                end = target.ids.index(target.info["end_of_turn"], shape)     # the end of the user turn
                spans = {"S": [shape], "A": list(range(shape + 1, end)), "SA": list(range(shape, end))}
                text_sources = range(shape, colon)
                binding, query_key = HEADS["binding"], HEADS["query_and_key"]
                patches = {f"outputs_{name}": at_tokens(source, target, "outputs", binding, tokens) for name, tokens in spans.items()}
                patches.update({f"queries_and_keys_{name}": at_tokens(source, target, "queries_and_keys", query_key, tokens)
                                for name, tokens in spans.items()})
                patches["outputs_SA_random"] = at_tokens(source, target, "outputs", random_like(binding, pair["index"]), spans["SA"])
                patches["queries_and_keys_SA_random"] = at_tokens(source, target, "queries_and_keys",
                                                                  random_like(query_key, pair["index"]), spans["SA"])
                run_S = record_patch(model, target, patches["outputs_S"])
                run_SA = record_patch(model, target, patches["outputs_SA"])
                patches.update({
                    "relay_from_S_run": relay_values(run_S, target, HEADS["relay"], text_sources),
                    "relay_from_SA_run": relay_values(run_SA, target, HEADS["relay"], text_sources),
                    "relay_from_SA_run_random": relay_values(run_SA, target, random_like(HEADS["relay"], pair["index"]), text_sources),
                    "relay_only_at_S": relay_values(run_SA, target, HEADS["relay"], spans["S"]),
                    "relay_only_at_A": relay_values(run_SA, target, HEADS["relay"], spans["A"]),
                    "relay_restored": at_tokens(source, target, "outputs", binding, spans["SA"]).add(
                        relay_values(target, target, HEADS["relay"], text_sources))})
                answers = dict(zip(patches, run_patches(model, target, list(patches.values()))["answers"]))
                results.add(pair["index"], **{f"{wording}_{name}": answer == " " + letters["T" if name == "relay_restored" else "K"]
                                              for name, answer in answers.items()})
    for column, wording in enumerate(WORDINGS):
        for name, paper in PAPER.items():
            results.report(f"{wording}_{name}", paper=paper[column])
    results.save()


if __name__ == "__main__":
    main()
