"""Find the language heads (Section 6.2).

The gaze heads return a concept. Something else must add the language of the question. To find
it we use the colored-shapes task. A source asks in language A about one image, and a target asks
in language B about another image. We track a word that neither run writes: the target's shape
in the source's language. For all 1,152 heads, one at a time, we copy the head's output at the
last prompt token from the source into the target, and score how much this raises the logit of
that word over the logit of the target's own word. Heads are ranked by their mean score over the
12 search images, their 3 sources each, and all 12 ordered pairs of languages (432 cases).

    python language/main_experiments/00_find_language_heads.py

The number of heads is not searched: we keep the top 16. The logits are read at the first token
of each word written after a space (see `shape_word_token`), as in the paper's search.

Paper: 16 language heads in layers 19-35, listed in utils/heads.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import LANGUAGES, color_cases, color_images, shape_word_token
from utils.experiment import Results, parse_args
from utils.heads import HEADS, rank_heads, score_each_head
from utils.model import load_model


def main():
    args = parse_args(__doc__, split="search", num_pairs=12)
    model = load_model(args.model, args.device)
    results = Results("find_language_heads", args)
    total, cases = 0, 0
    for spec in color_images("search", args.start + args.num_pairs)[args.start:]:
        for source, target, case in color_cases(model, spec, target_languages=LANGUAGES):
            moved = shape_word_token(model, case["source_language"], case["target_shape"])     # the language moved
            own = shape_word_token(model, case["target_language"], case["target_shape"])       # nothing changed
            total += score_each_head(model, source, target, "head_output", "last", lambda logits: logits[moved] - logits[own])
            cases += 1
        print(f"image {spec['index']} searched ({cases} cases)", flush=True)
    ranking = rank_heads(total / cases)
    print(f"language heads found: {ranking[:16]}")
    print(f"same as the paper's 16 language heads: {ranking[:16] == HEADS['language']}")
    results.save(found=ranking[:16], ranking=ranking, scores=(total / cases).tolist())


if __name__ == "__main__":
    main()
