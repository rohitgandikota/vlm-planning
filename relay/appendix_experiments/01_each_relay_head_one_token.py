"""Each relay head alone, reading one source token at a time (Appendix C.1, Figure "relay" b).

This extends relay/main_experiments/02_relay_one_token_at_a_time.py, which lets all 16 relay heads
read the source's value at one token. Here one relay head at a time, at the colon token, reads the
source's value at a single text token after the image. Its query, keys and attention weights stay
the target's own. We repeat this for every text token before the colon token.

Each cell of the figure is the mean gain of the logit of K over the logit of T, compared with the
unpatched target. K is the target's own letter at the panel the source points to, and T is the
target's own answer. The figure draws the shape token and the "?" after it as one column, the sum
of their two gains.

    python relay/appendix_experiments/01_each_relay_head_one_token.py --num_pairs 500

Paper: a map of gains, no single number. The row "all 16 relay heads" is the main experiment
(shape token 47.3%, "?" 9.9%). The map is saved in the result file under "gain".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import both_directions, one_answer_pairs
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import letter_token_ids, load_model
from utils.patching import make_patch, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    ids = letter_token_ids(model)
    results = Results("each_relay_head_one_token", args)
    heads = sorted(HEADS["relay"])
    total, cases = None, 0
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for source, target, letters in both_directions(model, pair):
            tokens = target.text_tokens()[:-1]                      # every text token before the colon token
            patches = [make_patch(source, target, "values", "colon", heads=[head], positions=[position])
                       for head in heads for position, _ in tokens]
            logits = run_patches(model, target, patches, max_batch=32)["logits"]
            k, t = ids[letters["K"]], ids[letters["T"]]
            unpatched = float(target.logits[k] - target.logits[t])
            gain = ((logits[:, k] - logits[:, t]) - unpatched).view(len(heads), len(tokens))
            total = gain if total is None else total + gain
            cases += 1
            results.add(pair["index"], shape_token_gain=gain[:, target.token("shape") - target.first_text].tolist())
    gain = (total / cases).tolist()
    names = ["shape token" if position == target.token("shape") else text for position, text in tokens]
    shape = names.index("shape token")
    print("mean gain of K over T when one relay head reads one token")
    print("head        shape token + '?'   largest other token")
    for (layer, head), row in zip(heads, gain):
        other = max(value for i, value in enumerate(row) if i not in (shape, shape + 1))
        print(f"L{layer}H{head:<8d} {row[shape] + row[shape + 1]:17.2f} {other:21.2f}")
    results.save(gain={"heads": heads, "tokens": names, "mean_gain": gain, "cases": cases})


if __name__ == "__main__":
    main()
