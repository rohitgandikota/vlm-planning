"""The attention heads found in the paper, and the tools to find them again.

Heads are (layer, head) pairs, both counted from 0. The lists below are for Qwen3-VL-8B-Instruct
(36 layers, 32 heads per layer), in the order of their search ranking.

    from utils.heads import HEADS, random_like
    HEADS["binding"]                               # the 16 binding heads
    random_like(HEADS["binding"], seed=0)          # 16 random heads from the same layers (the control)
"""
import random
from pathlib import Path

HEADS = {
    # Section 4.1. Binding heads: found by patching head outputs at the shape token.
    "binding": [(18, 15), (21, 11), (17, 4), (19, 27), (15, 9), (19, 24), (21, 18), (15, 11), (15, 2), (17, 7),
                (20, 29), (21, 8), (16, 27), (20, 15), (17, 15), (18, 11)],
    # Found by patching queries at the shape token (they carry the requested shape).
    "query": [(19, 27), (15, 9), (19, 24), (18, 15), (15, 11), (17, 4), (15, 10), (15, 2)],
    # Found by letting heads read the source's image keys at the shape token (they carry the layout).
    "key": [(19, 27), (15, 9), (19, 24), (15, 11), (18, 15), (15, 2), (21, 11), (18, 12)],
    # Section 4.2. Relay heads: found by patching the values read at the colon token.
    "relay": [(22, 10), (22, 11), (35, 25), (23, 26), (25, 15), (19, 21), (34, 28), (23, 0), (20, 2), (20, 3),
              (23, 3), (25, 12), (28, 22), (21, 19), (22, 3), (22, 0)],
}
# The query and key heads together (10 heads), and the union with the binding heads (18 heads).
HEADS["query_and_key"] = sorted(set(HEADS["query"]) | set(HEADS["key"]))
HEADS["binding_query_key"] = sorted(set(HEADS["binding"]) | set(HEADS["query_and_key"]))


def random_like(heads, seed, n_layers=36, n_heads=32, exclude=()):
    """The control used in the paper: one random head for every given head, from the same layer,
    never one of the given heads. Returns a sorted list."""
    rng = random.Random(seed)
    taken, banned = set(), set(heads) | set(exclude)
    for layer, _ in heads:
        options = [(layer, h) for h in range(n_heads) if (layer, h) not in banned and (layer, h) not in taken]
        if not options:                      # the layer is used up: take any free head
            options = [(l, h) for l in range(n_layers) for h in range(n_heads)
                       if (l, h) not in banned and (l, h) not in taken]
        taken.add(rng.choice(options))
    return sorted(taken)


def rank_heads(scores):
    """Sort heads by a score. `scores` is a tensor or array (layers, heads); the best head comes first."""
    flat = [(float(scores[layer][head]), layer, head) for layer in range(len(scores)) for head in range(len(scores[layer]))]
    return [(layer, head) for _, layer, head in sorted(flat, key=lambda x: -x[0])]


def choose_k(rates, threshold=0.8):
    """The paper's rule for the number of heads. `rates` maps K to the rate of the patch with the
    top K heads on validation pairs. Returns the smallest K that reaches the threshold, or the best
    K if none does."""
    passing = [k for k in sorted(rates) if rates[k] >= threshold]
    return passing[0] if passing else max(sorted(rates), key=lambda k: rates[k])


# ---------------------------------------------------------------------------------------------- published Gaze Heads, and the head search
HEADS.update({
    # The top 100 published Gaze Heads (Gandikota and Bau, 2026), in the order of their Gaze score.
    # They are never patched in Section 4. The head searches use them as a second readout: the
    # panel that these heads look at from the colon token (see `sweep_k`).
    "published_gaze": [
        (23, 10), (23, 30), (23, 28), (21, 18), (24, 13), (21, 11), (25, 11), (24, 29), (22, 4), (23, 13),
        (25, 10), (24, 23), (26, 26), (26, 18), (26, 31), (23, 4), (27, 16), (24, 8), (23, 12), (20, 15),
        (23, 11), (21, 10), (25, 30), (24, 14), (22, 5), (28, 16), (23, 31), (26, 24), (27, 18), (24, 15),
        (23, 5), (26, 27), (24, 31), (28, 2), (28, 3), (26, 15), (24, 19), (21, 23), (26, 17), (24, 16),
        (21, 31), (26, 29), (25, 9), (28, 19), (28, 23), (23, 14), (27, 29), (21, 16), (22, 2), (24, 12),
        (24, 26), (31, 4), (28, 0), (22, 0), (28, 11), (32, 8), (24, 30), (24, 5), (34, 28), (27, 17),
        (31, 30), (21, 28), (31, 28), (21, 25), (21, 27), (32, 3), (21, 19), (29, 27), (31, 5), (32, 0),
        (34, 7), (27, 19), (21, 24), (28, 9), (29, 6), (26, 25), (30, 27), (31, 3), (33, 12), (33, 6),
        (28, 18), (31, 6), (24, 28), (33, 29), (33, 13), (28, 8), (31, 15), (32, 2), (33, 30), (24, 21),
        (22, 11), (33, 10), (28, 10), (30, 6), (33, 15), (33, 23), (24, 11), (32, 11), (34, 1), (34, 21),
    ],
})

K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)      # the numbers of heads tried in the K sweep


def search_heads(model, pairs, searches, folder=None):
    """Step 1 of a head search: patch one head at a time, for every head of the model.

    searches   {name: (build, letter)}. `build(source, target, heads)` returns the patch for a list
               of heads, and `letter` is the outcome letter this patch should give ("K", "P" or "R").
    folder     if given, the scores of each pair are kept there, so a long search can be stopped,
               continued, or run in parts on several GPUs.

    The score of a head is how much its patch raises the logit of `letter` over the logit of the
    target's own letter T, compared with the unpatched target. Returns {name: tensor (layers, heads)},
    the mean score over the pairs and both directions."""
    import torch
    from utils.dataset import both_directions
    from utils.model import all_heads, letter_token_ids, n_heads, n_layers
    from utils.patching import run_patches
    ids, heads = letter_token_ids(model), all_heads(model)
    shape = (n_layers(model), n_heads(model))
    total = {name: torch.zeros(shape) for name in searches}
    for pair in pairs:
        file = Path(folder) / f"pair{pair['index']:04d}.pt" if folder else None
        if file and file.exists():
            scores = torch.load(file)
        else:
            scores = {name: torch.zeros(shape) for name in searches}
            for source, target, letters in both_directions(model, pair):
                own = ids[letters["T"]]
                for name, (build, letter) in searches.items():
                    want = ids[letters[letter]]
                    patches = [build(source, target, [head]) for head in heads]
                    logits = run_patches(model, target, patches, max_batch=32)["logits"]
                    unpatched = float(target.logits[want] - target.logits[own])
                    scores[name] += ((logits[:, want] - logits[:, own]) - unpatched).view(shape)
            if file:
                file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(scores, file)
        for name in searches:
            total[name] += scores[name]
        print(f"pair {pair['index']} searched", flush=True)
    return {name: total[name] / (2 * len(pairs)) for name in searches}


def sweep_k(model, pairs, ranking, build, letter, sizes=K_VALUES):
    """Step 2 of a head search: patch the top K heads of a ranking together, for every K in `sizes`.

    Returns {K: {"letter": rate, "gaze": rate}} over the pairs and both directions.
        letter   the next token is the outcome letter the patch should give
        gaze     the published Gaze Heads, which are not patched, look at the panel of that letter"""
    from utils.dataset import both_directions
    from utils.patching import panel_attention, run_patches
    hits = {k: {"letter": 0, "gaze": 0} for k in sizes}
    cases = 0
    for pair in pairs:
        for source, target, letters in both_directions(model, pair):
            patches = [build(source, target, ranking[:k]) for k in sizes]
            result = run_patches(model, target, patches, keep_attention=True)
            looked_at = panel_attention(result["attention"], HEADS["published_gaze"], target).argmax(dim=1)
            expected = target.image.letters.index(letters[letter])       # the panel that holds the letter
            for k, answer, panel in zip(sizes, result["answers"], looked_at.tolist()):
                hits[k]["letter"] += answer == " " + letters[letter]
                hits[k]["gaze"] += panel == expected
            cases += 1
    return {k: {name: count / cases for name, count in hits[k].items()} for k in sizes}


def select_heads(model, pairs, scores, build, letter, name=""):
    """Steps 2 and 3 of a head search: rank the heads by their score, run the K sweep on validation
    pairs, and take the smallest K for which both the letter rate and the Gaze rate reach 80%
    (the best K if none does). Returns (chosen heads, rates per K)."""
    ranking = rank_heads(scores)
    rates = sweep_k(model, pairs, ranking, build, letter)
    k = choose_k({k: min(rate["letter"], rate["gaze"]) for k, rate in rates.items()})
    print(f"\n{name}: K sweep on {len(pairs)} validation pairs (letter % / Gaze %)")
    for size, rate in rates.items():
        print(f"  K = {size:3d}   {100 * rate['letter']:5.1f} / {100 * rate['gaze']:5.1f}" + ("   <- chosen" if size == k else ""))
    return ranking[:k], rates


# ---------------------------------------------------------------------------------------------- answer-token heads
HEADS.update({
    # Section 5.3. Answer-token heads: found by letting heads read the source's image keys at the
    # answer token (progress/main_experiments/00_find_answer_token_heads.py). No number of heads
    # reached 80% on the validation pairs, so the rule took the best one, 32.
    "answer_token": [(19, 27), (19, 24), (20, 31), (21, 11), (15, 9), (19, 25), (19, 12), (20, 30), (24, 29), (20, 28),
                     (19, 13), (19, 26), (15, 11), (20, 13), (23, 30), (21, 8), (22, 0), (19, 14), (34, 12), (20, 19),
                     (29, 11), (20, 18), (16, 31), (34, 22), (33, 31), (15, 8), (21, 19), (21, 23), (21, 9), (22, 11),
                     (24, 31), (20, 4)],
})


# ---------------------------------------------------------------------------------------------- stop heads
# Section 5.4. The search at the last answer token of a list (progress/main_experiments/00_find_stop_heads.py)
# ranks every head by how much its output moves the choice between a comma and the end of the turn.
# These are the best 128 heads of that ranking, best first. The first 64 are the stop heads. The
# random control of the paper draws its heads outside all 128.
STOP_HEAD_RANKING = [
    (26, 5), (34, 4), (21, 19), (34, 1), (32, 24), (35, 2), (20, 2), (23, 27), (22, 25), (23, 26), (22, 10),
    (21, 13), (32, 26), (34, 8), (28, 26), (23, 30), (23, 3), (32, 8), (28, 29), (23, 2), (21, 0), (35, 27),
    (29, 9), (34, 31), (28, 27), (21, 1), (22, 27), (20, 30), (22, 11), (23, 0), (27, 12), (23, 25), (33, 7),
    (18, 8), (22, 26), (20, 8), (20, 31), (32, 15), (29, 14), (20, 3), (32, 25), (21, 25), (25, 12), (25, 5),
    (21, 30), (30, 7), (33, 0), (25, 13), (22, 29), (21, 14), (20, 28), (20, 11), (23, 12), (21, 11), (24, 28),
    (22, 0), (20, 10), (34, 13), (35, 7), (24, 2), (26, 14), (17, 30), (34, 26), (23, 5),
    (27, 19), (24, 29), (32, 3), (32, 21), (30, 4), (19, 13), (23, 28), (18, 9), (33, 29), (22, 17), (33, 9),
    (24, 11), (22, 30), (24, 22), (22, 18), (21, 18), (22, 7), (18, 15), (25, 16), (27, 16), (22, 24), (19, 27),
    (35, 13), (21, 10), (31, 15), (19, 19), (14, 8), (31, 21), (32, 14), (19, 18), (35, 5), (25, 30), (34, 28),
    (35, 15), (21, 20), (18, 3), (34, 9), (15, 18), (18, 0), (27, 2), (32, 16), (35, 28), (31, 29), (34, 10),
    (21, 8), (24, 27), (19, 14), (18, 20), (27, 23), (35, 14), (34, 23), (18, 29), (31, 16), (30, 6), (25, 28),
    (20, 17), (19, 24), (30, 0), (29, 1), (19, 22), (28, 16), (21, 15), (18, 28), (22, 28)]

HEADS.update({
    # Stop heads: their outputs at the last answer token decide whether the list goes on or stops.
    "stop": STOP_HEAD_RANKING[:64],
    # The 32 heads found at the shape token of the list task, by patching head outputs there so that
    # the first answer moves. Used when a source's set of matching panels is patched (Appendix E).
    "list_shape_token": [(18, 11), (18, 15), (19, 24), (19, 27), (21, 11), (19, 14), (17, 4), (19, 13), (18, 12),
                         (20, 29), (16, 27), (17, 15), (21, 18), (17, 12), (21, 19), (14, 30), (20, 17), (19, 21),
                         (20, 5), (20, 2), (15, 9), (15, 11), (20, 15), (21, 8), (18, 23), (21, 10), (19, 26),
                         (19, 3), (20, 8), (20, 16), (18, 21), (19, 31)],
})


# ---------------------------------------------------------------------------------------------- gaze and language heads
HEADS.update({
    # Section 6.1. Gaze heads: found by patching queries at the colon token (gaze/main_experiments/00).
    "gaze": [(24, 29), (34, 7), (23, 4), (24, 31), (29, 5), (24, 23), (26, 26), (29, 6), (22, 0), (22, 11), (21, 18),
             (24, 5), (21, 11), (33, 31), (32, 10), (29, 11), (24, 30), (32, 8), (23, 30), (28, 0), (34, 28), (32, 11),
             (27, 18), (29, 12), (21, 8), (24, 13), (29, 8), (30, 6), (20, 15), (29, 10), (21, 19), (19, 3)],
    # Section 6.2. Language heads: found by patching head outputs at the last prompt token of the
    # colored-shapes task (language/main_experiments/00).
    "language": [(24, 27), (32, 17), (22, 31), (20, 27), (30, 20), (31, 2), (35, 25), (35, 26), (31, 24), (31, 26),
                 (34, 15), (22, 19), (22, 8), (24, 25), (35, 5), (19, 21)],
    # The gaze and language heads of an earlier search on fewer pairs. The two COCO experiments of the
    # paper were run with these (31 of 32 and 13 of 16 heads are shared with the lists above).
    "gaze_coco": [(24, 29), (34, 7), (23, 4), (24, 31), (29, 5), (24, 23), (26, 26), (29, 6), (22, 0), (22, 11),
                  (21, 18), (21, 11), (33, 31), (32, 10), (29, 11), (24, 5), (24, 30), (32, 8), (28, 0), (34, 28),
                  (23, 30), (32, 11), (27, 18), (29, 12), (21, 8), (24, 13), (30, 6), (29, 8), (20, 15), (29, 10),
                  (21, 27), (21, 19)],
    "language_coco": [(24, 27), (32, 17), (22, 31), (35, 26), (35, 25), (30, 20), (31, 2), (33, 7), (31, 24), (22, 19),
                      (22, 14), (24, 25), (31, 26), (34, 12), (19, 21), (34, 15)],
})


def score_each_head(model, source, target, component, token, score):
    """Patch one head at a time and score the result. Used by the head searches.

    For every head of the model, `component` ("query", "head_output", ...) of that head alone is
    copied from the source into the target at `token`. `score(logits)` turns the next-token logits
    of one patched run into a number. Returns the gain of the score over the unpatched target, as a
    tensor (layers, heads)."""
    import torch
    from utils.model import n_heads, n_layers
    from utils.patching import make_patch, run_patches
    gains = torch.zeros(n_layers(model), n_heads(model))
    for layer in range(n_layers(model)):
        patches = [make_patch(source, target, component, token, heads=[(layer, head)]) for head in range(n_heads(model))]
        logits = run_patches(model, target, patches, max_batch=32)["logits"]
        gains[layer] = torch.stack([score(row) for row in logits]) - score(target.logits.cpu())
    return gains


# ==================================================================================================
# Appendix A and B
# ==================================================================================================
HEADS.update({
    # Appendix B.2. Query heads ranked by how much their query moves the request minus how much it
    # moves the pointer (binding/appendix_experiments/03). The rule took 128 heads. They come from an
    # earlier, smaller run on other pairs, in the order of that ranking.
    "query_without_pointer": [
    (19, 24), (19, 27), (21, 31), (19, 25), (21, 26), (15, 9), (20, 4), (20, 18), (18, 13), (18, 14), (15, 11),
    (15, 2), (19, 12), (15, 8), (17, 12), (18, 21), (19, 26), (21, 9), (17, 6), (17, 24), (21, 17), (20, 9),
    (19, 2), (22, 2), (18, 23), (22, 8), (13, 9), (20, 19), (17, 27), (20, 13), (15, 10), (16, 18), (18, 31),
    (17, 5), (17, 29), (14, 12), (21, 20), (23, 28), (14, 16), (11, 11), (18, 22), (18, 6), (16, 5), (20, 3),
    (16, 29), (15, 0), (20, 7), (17, 2), (9, 17), (13, 26), (16, 7), (18, 9), (9, 16), (12, 25), (19, 13),
    (17, 13), (22, 20), (19, 23), (16, 30), (7, 26), (15, 17), (20, 12), (17, 26), (13, 27), (18, 16),
    (12, 30), (16, 20), (21, 29), (16, 8), (24, 28), (22, 9), (13, 10), (17, 14), (19, 0), (3, 12), (15, 23),
    (18, 0), (21, 10), (15, 18), (14, 9), (24, 12), (12, 3), (21, 16), (11, 19), (5, 7), (16, 11), (20, 1),
    (17, 20), (11, 23), (18, 20), (16, 0), (16, 19), (13, 0), (18, 10), (16, 1), (19, 18), (20, 20), (9, 3),
    (16, 9), (19, 9), (12, 8), (11, 13), (23, 9), (10, 11), (15, 6), (16, 25), (18, 18), (13, 4), (13, 8),
    (14, 31), (14, 6), (10, 14), (3, 26), (14, 22), (18, 5), (15, 24), (14, 21), (18, 7), (14, 15), (15, 25),
    (14, 27), (13, 16), (6, 11), (22, 6), (17, 25), (13, 14), (23, 24), (9, 18)],
})


# ==================================================================================================
# Appendix D
# ==================================================================================================
HEADS.update({
    # Appendix D.3, Table "many heads". Heads found by the searches on the list task (same three steps
    # as progress/main_experiments/00_find_answer_token_heads.py, with other patches), best first.
    # Found by patching the values that the colon token reads from the question, for the first answer (16 heads).
    "list_colon_values": [
        (25, 15), (21, 19), (22, 10), (22, 11), (23, 26), (35, 25), (23, 0), (34, 28), (20, 29), (19, 21), (30, 5),
        (20, 2), (19, 31), (20, 8), (21, 25), (21, 30)],
    # Found by letting heads read the source's image keys at the shape token, for the first answer (128 heads).
    "list_shape_token_keys": [
        (19, 24), (18, 15), (13, 16), (15, 9), (15, 11), (19, 14), (17, 21), (19, 13), (18, 11), (18, 12), (17, 15),
        (16, 28), (17, 1), (21, 11), (18, 13), (21, 18), (15, 8), (13, 10), (20, 29), (19, 26), (19, 27), (15, 2),
        (15, 29), (18, 23), (17, 4), (18, 7), (20, 28), (15, 13), (20, 17), (16, 23), (17, 24), (16, 22), (19, 3),
        (18, 19), (13, 8), (15, 0), (20, 30), (16, 6), (14, 14), (13, 11), (9, 17), (20, 0), (19, 30), (18, 8),
        (16, 12), (19, 31), (12, 24), (21, 8), (15, 1), (17, 5), (17, 26), (14, 5), (18, 21), (14, 17), (18, 17),
        (15, 5), (13, 14), (18, 5), (17, 12), (16, 13), (11, 11), (15, 31), (8, 2), (18, 28), (21, 23), (17, 13),
        (21, 30), (18, 27), (20, 31), (16, 19), (18, 30), (11, 9), (15, 16), (18, 18), (16, 17), (9, 14), (17, 0),
        (18, 26), (13, 9), (10, 26), (21, 24), (14, 11), (13, 1), (15, 10), (19, 11), (21, 10), (14, 15), (18, 22),
        (19, 0), (12, 28), (22, 0), (13, 19), (13, 0), (19, 29), (18, 24), (15, 12), (21, 13), (21, 25), (9, 15),
        (18, 9), (16, 9), (11, 10), (21, 20), (15, 7), (19, 28), (17, 9), (14, 24), (13, 15), (13, 5), (14, 28),
        (17, 17), (8, 6), (6, 12), (9, 12), (14, 19), (13, 27), (11, 24), (20, 9), (11, 14), (12, 26), (13, 3),
        (22, 11), (20, 18), (16, 24), (20, 21), (21, 19), (8, 4), (12, 7)],
    # Found by patching queries at the colon token, for the first answer (128 heads).
    "list_colon_queries": [
        (24, 29), (24, 31), (29, 5), (23, 30), (34, 7), (29, 6), (24, 23), (29, 11), (28, 0), (23, 4), (26, 26),
        (33, 31), (21, 18), (27, 18), (21, 11), (32, 10), (24, 13), (21, 19), (32, 11), (29, 12), (21, 25), (22, 11),
        (22, 0), (21, 8), (32, 8), (34, 28), (30, 6), (20, 28), (20, 31), (32, 31), (20, 2), (21, 9), (24, 5),
        (21, 10), (29, 10), (21, 15), (24, 30), (20, 29), (20, 15), (27, 17), (29, 7), (31, 28), (25, 15), (33, 25),
        (31, 29), (24, 2), (30, 7), (28, 22), (28, 16), (23, 13), (21, 22), (24, 17), (22, 10), (31, 3), (21, 27),
        (35, 25), (32, 3), (23, 12), (21, 13), (19, 14), (26, 18), (35, 26), (35, 8), (20, 20), (18, 9), (22, 8),
        (21, 1), (31, 6), (23, 11), (33, 14), (35, 29), (26, 24), (21, 23), (34, 20), (34, 31), (20, 21), (19, 6),
        (34, 1), (20, 6), (35, 24), (19, 21), (23, 15), (18, 15), (30, 5), (20, 22), (28, 21), (35, 5), (26, 4),
        (22, 19), (18, 22), (18, 19), (33, 26), (18, 14), (21, 6), (19, 11), (34, 21), (20, 11), (28, 18), (33, 28),
        (19, 16), (18, 5), (33, 16), (24, 22), (25, 11), (33, 15), (24, 9), (29, 8), (26, 3), (33, 19), (20, 26),
        (26, 20), (28, 19), (25, 17), (22, 22), (26, 17), (20, 27), (33, 1), (31, 9), (31, 13), (35, 15), (18, 10),
        (26, 6), (23, 22), (32, 24), (29, 30), (35, 16), (20, 10), (24, 15)],
    # Found by patching queries at the last comma token, for the next answer (16 heads).
    "list_comma_queries": [
        (24, 29), (34, 7), (29, 5), (24, 31), (29, 11), (29, 6), (28, 0), (26, 26), (33, 31), (24, 30), (27, 18),
        (24, 23), (32, 10), (23, 4), (32, 11), (35, 8)],

    # Appendix D.4. These two come from an earlier, smaller run on 100 search pairs and 50 validation
    # pairs drawn from another seed. First, the heads ranked by how much the source's values of the
    # answer token, read at the comma token by one head, move the next answer (the best 128).
    "comma_reads_answer_token": [
        (35, 25), (22, 10), (19, 23), (22, 30), (34, 28), (20, 1), (22, 24), (23, 0), (27, 12), (21, 17), (23, 24),
        (21, 1), (21, 14), (25, 15), (20, 4), (22, 2), (20, 0), (19, 2), (23, 29), (19, 20), (23, 23), (35, 26),
        (25, 12), (23, 3), (25, 13), (19, 22), (18, 14), (30, 7), (30, 5), (20, 13), (34, 20), (23, 31), (23, 7),
        (28, 28), (28, 22), (21, 0), (21, 29), (22, 1), (19, 17), (20, 8), (22, 26), (21, 26), (21, 16), (20, 11),
        (20, 7), (25, 31), (20, 18), (30, 4), (16, 18), (21, 31), (23, 2), (18, 12), (21, 22), (21, 3), (22, 16),
        (35, 16), (21, 10), (30, 23), (31, 24), (32, 14), (18, 3), (30, 6), (22, 21), (34, 21), (17, 22), (19, 9),
        (19, 11), (19, 25), (21, 13), (28, 21), (34, 17), (24, 31), (31, 9), (19, 18), (17, 12), (29, 6), (14, 11),
        (35, 11), (22, 25), (25, 17), (22, 11), (23, 28), (18, 31), (16, 25), (18, 24), (20, 10), (17, 30), (14, 10),
        (35, 2), (14, 31), (31, 29), (28, 4), (35, 15), (19, 0), (19, 30), (25, 6), (17, 27), (20, 26), (33, 25),
        (26, 12), (15, 30), (22, 15), (26, 11), (19, 12), (34, 8), (32, 27), (19, 7), (14, 9), (28, 1), (35, 17),
        (27, 4), (27, 18), (17, 5), (16, 4), (24, 30), (24, 29), (20, 6), (26, 5), (24, 22), (25, 0), (18, 5),
        (26, 26), (16, 30), (23, 20), (19, 15), (32, 22), (28, 27), (24, 7)],
    # The 8 heads found by patching the answer token's values read at the comma token, with the comma
    # token blocked from the image except through the gaze heads (no K reached 80%; the best was 8).
    "value_route": [(22, 10), (20, 1), (20, 4), (20, 0), (22, 30), (18, 14), (35, 25), (20, 18)],
})


# ==================================================================================================
# Appendix E
# ==================================================================================================
# Appendix E.1. The search at the last answer token of a list also ranked every head by the values
# it reads from the prompt there (shape token through colon token), scored like the stop heads.
# These are the best 128 heads of that ranking, best first. No smaller set reached 80% on the
# validation pairs before 64, so the first 64 are used. The random control draws outside all 128.
STOP_VALUES_RANKING = [
    (21, 19), (23, 26), (22, 10), (23, 2), (21, 13), (21, 0), (22, 25), (20, 3), (21, 1), (23, 3), (25, 13),
    (28, 26), (28, 27), (22, 27), (22, 11), (21, 30), (23, 0), (20, 8), (20, 11), (20, 2), (32, 24), (25, 12),
    (22, 7), (25, 16), (20, 10), (21, 25), (32, 26), (28, 29), (18, 8), (34, 8), (34, 13), (22, 17), (15, 18),
    (21, 14), (14, 8), (18, 9), (23, 25), (22, 18), (27, 12), (22, 1), (31, 21), (19, 22), (25, 5), (18, 3),
    (19, 19), (30, 4), (19, 18), (27, 2), (19, 6), (19, 21), (17, 6), (17, 28), (22, 0), (26, 9), (18, 21),
    (15, 25), (23, 30), (18, 28), (33, 27), (23, 19), (19, 1), (17, 2), (27, 1), (23, 27),
    (16, 29), (18, 29), (19, 31), (18, 10), (25, 4), (22, 29), (29, 30), (20, 12), (15, 23), (33, 26), (17, 31),
    (34, 10), (28, 30), (27, 9), (16, 4), (18, 20), (16, 8), (14, 18), (19, 3), (20, 29), (22, 14), (16, 19),
    (24, 27), (26, 5), (32, 25), (18, 0), (22, 30), (25, 2), (32, 16), (22, 3), (35, 13), (23, 31), (35, 2),
    (28, 6), (26, 3), (25, 17), (19, 4), (19, 5), (35, 7), (20, 17), (26, 29), (30, 23), (33, 19), (34, 30),
    (20, 14), (30, 7), (28, 4), (19, 20), (31, 10), (21, 27), (21, 20), (25, 27), (34, 6), (18, 24), (32, 20),
    (20, 19), (22, 24), (21, 6), (17, 26), (30, 3), (33, 0), (29, 20), (15, 19), (23, 20)]

HEADS.update({
    # Appendix E.1. Heads whose values at the last answer token decide whether the list goes on.
    "stop_values": STOP_VALUES_RANKING[:64],
    # Appendix E.7, first search. Head outputs at the last comma token, layers 20-25, ranked for the
    # next panel on an earlier, smaller run (progress/appendix_experiments/00_find_selection_heads.py).
    # No size reached 80%; the best size, 64, is used. Best first.
    "selection_output": [
        (25, 15), (21, 19), (21, 11), (22, 10), (21, 18), (20, 30), (24, 29), (25, 13), (20, 29), (23, 26), (23, 30),
        (24, 13), (24, 31), (20, 8), (24, 30), (21, 1), (20, 31), (22, 24), (21, 8), (22, 11), (23, 0), (20, 11),
        (21, 14), (22, 30), (20, 1), (23, 3), (25, 12), (20, 3), (20, 10), (24, 23), (24, 21), (22, 28), (20, 15),
        (20, 0), (22, 1), (23, 29), (21, 6), (22, 0), (21, 24), (21, 27), (21, 22), (23, 23), (22, 25), (21, 9),
        (20, 18), (21, 0), (22, 16), (20, 14), (25, 6), (22, 5), (22, 4), (20, 17), (20, 21), (21, 16), (21, 15),
        (22, 29), (22, 22), (22, 19), (24, 24), (25, 17), (24, 17), (24, 4), (20, 16), (24, 14)],
    # Appendix E.7, the two searches with no source, on the same earlier run
    # (progress/appendix_experiments/00_find_selection_rankings.py). The best 64 of each ranking, best first.
    # By zeroing: the drop of the margin of the next letter over the other matching letters.
    "selection_by_zeroing": [
        (21, 11), (20, 31), (23, 26), (23, 3), (19, 13), (21, 18), (21, 19), (25, 5), (20, 30), (22, 28), (20, 29),
        (18, 14), (20, 1), (22, 10), (19, 31), (22, 18), (23, 2), (24, 27), (22, 13), (20, 11), (23, 30), (19, 23),
        (20, 8), (23, 29), (20, 2), (21, 8), (22, 19), (24, 21), (22, 25), (22, 26), (19, 14), (18, 8), (21, 25),
        (19, 3), (24, 29), (20, 10), (23, 5), (20, 12), (21, 9), (23, 0), (22, 0), (21, 15), (18, 10), (25, 31),
        (21, 20), (20, 0), (20, 15), (18, 28), (20, 14), (21, 3), (22, 4), (25, 15), (16, 6), (19, 1), (22, 20),
        (16, 16), (23, 12), (23, 28), (18, 9), (20, 23), (18, 22), (24, 8), (21, 30), (18, 4)],
    # By attention shift: share of attention on the next panel minus the mean share on the other matching panels.
    "selection_by_attention_shift": [
        (23, 4), (20, 29), (24, 29), (20, 30), (21, 11), (24, 23), (21, 10), (21, 8), (18, 15), (24, 5), (23, 5),
        (23, 30), (24, 13), (21, 26), (21, 18), (25, 30), (24, 31), (24, 8), (21, 9), (24, 30), (24, 14), (21, 31),
        (23, 31), (20, 21), (21, 16), (23, 10), (24, 26), (17, 15), (24, 9), (20, 18), (24, 12), (22, 22), (22, 0),
        (23, 28), (18, 14), (24, 28), (21, 25), (24, 15), (21, 27), (24, 7), (21, 19), (22, 4), (19, 25), (22, 11),
        (21, 23), (24, 22), (24, 11), (24, 21), (25, 11), (22, 20), (22, 2), (24, 20), (24, 4), (19, 31), (20, 16),
        (25, 28), (19, 27), (24, 19), (20, 31), (23, 12), (16, 23), (23, 13), (22, 21), (18, 21)],
})


# ==================================================================================================
# Appendix F
# ==================================================================================================
HEADS.update({
    # Appendix F.1. The top 256 heads of the gaze-head search (query patch at the colon token), best
    # first. The first 32 are HEADS["gaze"].
    "gaze_ranking": [
        (24, 29), (34, 7), (23, 4), (24, 31), (29, 5), (24, 23), (26, 26), (29, 6), (22, 0), (22, 11), (21, 18),
        (24, 5), (21, 11), (33, 31), (32, 10), (29, 11), (24, 30), (32, 8), (23, 30), (28, 0), (34, 28), (32, 11),
        (27, 18), (29, 12), (21, 8), (24, 13), (29, 8), (30, 6), (20, 15), (29, 10), (21, 19), (19, 3), (21, 27),
        (21, 24), (22, 10), (18, 15), (21, 25), (20, 2), (25, 15), (32, 31), (28, 21), (21, 16), (31, 29), (27, 17),
        (20, 29), (28, 22), (25, 12), (23, 26), (23, 7), (21, 30), (26, 24), (19, 13), (21, 13), (34, 31), (20, 10),
        (31, 3), (22, 22), (27, 19), (19, 14), (35, 25), (23, 2), (23, 3), (33, 14), (26, 20), (31, 6), (31, 15),
        (32, 3), (33, 28), (28, 16), (24, 19), (20, 21), (24, 22), (22, 14), (22, 1), (18, 30), (25, 13), (22, 19),
        (24, 20), (20, 11), (34, 1), (33, 16), (18, 10), (19, 1), (21, 23), (20, 16), (20, 14), (19, 29), (26, 31),
        (22, 4), (29, 7), (33, 19), (19, 31), (18, 23), (22, 3), (30, 4), (28, 18), (17, 6), (23, 0), (20, 6),
        (22, 13), (24, 9), (24, 26), (21, 4), (23, 15), (20, 30), (20, 0), (21, 28), (24, 15), (22, 18), (31, 13),
        (27, 12), (19, 27), (24, 16), (21, 6), (26, 18), (23, 14), (34, 20), (26, 29), (19, 10), (23, 13), (21, 1),
        (26, 19), (22, 7), (21, 14), (17, 13), (25, 11), (23, 21), (20, 17), (24, 0), (21, 22), (26, 22), (17, 3),
        (17, 24), (24, 27), (20, 28), (18, 29), (29, 27), (24, 1), (26, 7), (18, 12), (33, 7), (23, 11), (33, 6),
        (26, 17), (23, 12), (33, 13), (29, 1), (18, 28), (34, 30), (18, 27), (25, 5), (18, 24), (19, 8), (30, 1),
        (31, 9), (24, 10), (22, 30), (20, 8), (19, 19), (28, 7), (31, 8), (28, 19), (34, 21), (34, 17), (33, 8),
        (29, 14), (25, 26), (35, 21), (24, 8), (17, 12), (35, 29), (28, 4), (29, 17), (20, 5), (20, 31), (30, 23),
        (23, 27), (18, 20), (32, 14), (32, 29), (17, 26), (23, 25), (23, 10), (17, 18), (30, 20), (25, 6), (22, 21),
        (30, 15), (21, 7), (31, 10), (17, 2), (35, 12), (28, 28), (22, 27), (20, 3), (22, 5), (15, 13), (28, 6),
        (26, 0), (29, 26), (33, 22), (24, 25), (28, 15), (17, 7), (34, 13), (18, 19), (29, 0), (33, 27), (23, 20),
        (34, 16), (17, 25), (25, 29), (19, 21), (16, 19), (35, 11), (31, 5), (22, 17), (25, 24), (27, 23), (34, 19),
        (23, 16), (24, 6), (16, 25), (26, 14), (19, 16), (17, 23), (32, 15), (19, 0), (30, 10), (27, 4), (35, 20),
        (23, 18), (22, 23), (28, 29), (30, 2), (22, 25), (35, 28), (31, 17), (35, 19), (28, 5), (15, 10), (23, 22),
        (18, 13), (29, 3), (29, 24), (16, 28), (35, 23), (32, 30), (16, 3), (16, 4), (15, 17), (34, 27), (35, 16),
        (32, 19), (19, 11), (27, 30)],
    # Appendix F.1. The heads of the second search: one head at a time is blocked from the image at
    # the colon token. No size reaches 80%, so the rule takes the best one, 8 heads.
    "blocking_search": [
        (24, 28), (18, 14), (21, 23), (21, 11), (20, 15), (31, 31), (21, 18), (22, 2)],
    # The top 256 published Gaze Heads (Gandikota and Bau, 2026), in the order of their Gaze score.
    # The first 100 are HEADS["published_gaze"].
    "published_gaze_ranking": [
        (23, 10), (23, 30), (23, 28), (21, 18), (24, 13), (21, 11), (25, 11), (24, 29), (22, 4), (23, 13), (25, 10),
        (24, 23), (26, 26), (26, 18), (26, 31), (23, 4), (27, 16), (24, 8), (23, 12), (20, 15), (23, 11), (21, 10),
        (25, 30), (24, 14), (22, 5), (28, 16), (23, 31), (26, 24), (27, 18), (24, 15), (23, 5), (26, 27), (24, 31),
        (28, 2), (28, 3), (26, 15), (24, 19), (21, 23), (26, 17), (24, 16), (21, 31), (26, 29), (25, 9), (28, 19),
        (28, 23), (23, 14), (27, 29), (21, 16), (22, 2), (24, 12), (24, 26), (31, 4), (28, 0), (22, 0), (28, 11),
        (32, 8), (24, 30), (24, 5), (34, 28), (27, 17), (31, 30), (21, 28), (31, 28), (21, 25), (21, 27), (32, 3),
        (21, 19), (29, 27), (31, 5), (32, 0), (34, 7), (27, 19), (21, 24), (28, 9), (29, 6), (26, 25), (30, 27),
        (31, 3), (33, 12), (33, 6), (28, 18), (31, 6), (24, 28), (33, 29), (33, 13), (28, 8), (31, 15), (32, 2),
        (33, 30), (24, 21), (22, 11), (33, 10), (28, 10), (30, 6), (33, 15), (33, 23), (24, 11), (32, 11), (34, 1),
        (34, 21), (26, 19), (33, 14), (33, 19), (34, 0), (29, 11), (31, 31), (21, 26), (30, 25), (33, 18), (33, 16),
        (29, 9), (27, 13), (30, 17), (26, 20), (33, 28), (34, 5), (31, 13), (18, 15), (30, 9), (34, 2), (26, 4),
        (32, 5), (20, 30), (27, 31), (24, 7), (35, 22), (21, 29), (29, 12), (31, 14), (21, 8), (20, 4), (26, 30),
        (29, 26), (33, 9), (31, 9), (18, 30), (30, 8), (20, 13), (29, 13), (33, 17), (32, 10), (31, 7), (31, 29),
        (33, 11), (26, 16), (20, 16), (29, 5), (21, 9), (30, 24), (32, 7), (21, 20), (20, 29), (29, 25), (30, 26),
        (34, 19), (2, 17), (0, 27), (21, 17), (29, 7), (30, 11), (27, 15), (34, 3), (20, 28), (20, 21), (29, 0),
        (29, 15), (21, 30), (18, 14), (29, 3), (33, 31), (33, 21), (0, 5), (20, 31), (29, 24), (19, 14), (24, 22),
        (20, 23), (32, 6), (24, 9), (5, 29), (19, 12), (20, 18), (21, 22), (33, 20), (18, 23), (28, 21), (25, 8),
        (31, 20), (29, 1), (18, 27), (3, 20), (0, 8), (26, 14), (25, 26), (25, 23), (23, 26), (29, 4), (17, 17),
        (22, 22), (2, 1), (34, 31), (12, 18), (19, 13), (25, 28), (27, 14), (34, 30), (23, 29), (27, 30), (20, 17),
        (0, 29), (17, 24), (10, 22), (17, 27), (34, 17), (23, 8), (28, 12), (28, 22), (28, 1), (29, 8), (27, 28),
        (23, 21), (21, 4), (29, 2), (22, 17), (22, 20), (18, 21), (17, 26), (18, 22), (18, 29), (23, 9), (23, 23),
        (25, 24), (18, 12), (31, 2), (24, 17), (20, 20), (28, 17), (18, 5), (22, 6), (27, 7), (18, 19), (25, 20),
        (5, 2), (1, 1), (20, 19), (17, 4), (24, 4), (26, 7), (35, 19), (28, 13), (17, 5), (33, 22), (32, 1),
        (26, 6), (24, 25), (17, 15)],
    # Appendix F.2. The top 100 heads by the Gaze score measured on the one-answer task: the attention
    # a head puts from the colon token on the asked panel (gaze/appendix_experiments/03). From an
    # earlier set of 250 pairs (2,000 questions).
    "gaze_score_on_this_task": [
        (23, 4), (21, 10), (21, 11), (21, 18), (23, 30), (21, 8), (23, 5), (29, 11), (24, 29), (24, 23), (26, 26),
        (28, 0), (24, 5), (21, 16), (20, 15), (24, 31), (23, 10), (23, 28), (18, 15), (23, 31), (24, 26), (22, 0),
        (21, 23), (21, 19), (29, 8), (21, 31), (32, 10), (29, 5), (24, 13), (32, 8), (31, 28), (31, 31), (21, 27),
        (24, 7), (28, 23), (21, 9), (21, 24), (25, 30), (21, 26), (20, 13), (26, 18), (26, 24), (20, 21), (24, 28),
        (22, 4), (32, 11), (29, 12), (19, 14), (29, 6), (19, 13), (24, 12), (22, 2), (29, 7), (20, 23), (21, 25),
        (20, 29), (26, 20), (29, 15), (20, 18), (31, 30), (22, 11), (24, 4), (27, 18), (18, 23), (28, 16), (29, 9),
        (33, 13), (33, 28), (33, 31), (19, 12), (19, 3), (23, 11), (18, 27), (24, 19), (29, 10), (18, 22), (21, 20),
        (24, 14), (25, 11), (33, 29), (29, 13), (21, 17), (34, 7), (20, 16), (34, 5), (31, 6), (23, 14), (26, 25),
        (20, 30), (27, 16), (0, 27), (33, 14), (24, 15), (22, 6), (31, 15), (33, 12), (25, 10), (20, 31), (31, 13),
        (26, 19)],
    # The 16 heads found on the list task by patching queries at the last comma token (Appendix E).
    "list_last_comma_queries": [
        (24, 29), (34, 7), (29, 5), (24, 31), (29, 11), (29, 6), (28, 0), (26, 26), (33, 31), (24, 30), (27, 18),
        (24, 23), (32, 10), (23, 4), (32, 11), (35, 8)],
    # Appendix F.4. The top 128 heads of the concept search on the colored-shapes task: one head's
    # output at the last prompt token is copied from the source, and the score is the gain of the
    # source's shape over the target's shape, both in the target's language (gaze/appendix_experiments/08).
    "concept_search": [
        (26, 20), (22, 10), (24, 5), (24, 29), (23, 26), (25, 15), (26, 26), (29, 6), (26, 18), (23, 30), (31, 6),
        (24, 31), (22, 11), (35, 22), (27, 18), (30, 7), (26, 24), (30, 9), (32, 3), (34, 0), (23, 10), (34, 28),
        (24, 23), (29, 27), (26, 19), (28, 0), (19, 21), (24, 13), (23, 0), (25, 12), (24, 30), (21, 18), (23, 3),
        (29, 11), (22, 0), (34, 21), (33, 17), (27, 12), (30, 5), (27, 8), (29, 10), (35, 23), (24, 19), (35, 26),
        (20, 2), (34, 17), (33, 6), (20, 3), (31, 26), (35, 25), (31, 13), (25, 13), (28, 4), (21, 19), (29, 1),
        (29, 12), (22, 1), (20, 8), (21, 11), (32, 2), (27, 19), (35, 19), (28, 11), (21, 25), (33, 30), (20, 29),
        (23, 25), (26, 31), (26, 17), (23, 4), (28, 16), (28, 28), (30, 6), (28, 22), (22, 12), (28, 6), (25, 26),
        (26, 27), (22, 3), (20, 10), (35, 2), (31, 4), (27, 17), (21, 27), (31, 9), (23, 11), (31, 5), (17, 6),
        (27, 29), (19, 31), (20, 15), (34, 5), (24, 27), (27, 28), (19, 3), (28, 5), (23, 13), (21, 8), (27, 31),
        (33, 20), (21, 30), (28, 21), (24, 16), (26, 22), (20, 11), (20, 5), (19, 23), (35, 8), (18, 15), (33, 23),
        (20, 27), (29, 26), (30, 8), (20, 30), (34, 12), (30, 25), (31, 15), (34, 23), (22, 4), (21, 14), (33, 19),
        (25, 17), (34, 31), (24, 24), (27, 4), (29, 13), (18, 10), (23, 2)],
    # Appendix F.5. The same search on the object task (an earlier run on 12 images).
    "concept_search_objects": [
        (29, 27), (22, 10), (24, 29), (26, 20), (22, 11), (27, 18), (35, 26), (24, 31), (25, 15), (29, 6), (35, 25),
        (34, 28), (24, 23), (31, 3), (23, 26), (26, 17), (30, 7), (22, 0), (32, 8), (26, 26), (23, 30), (31, 13),
        (31, 6), (24, 5), (28, 4), (35, 22), (28, 11), (24, 13), (23, 10), (32, 11), (33, 9), (29, 25), (32, 3),
        (34, 1), (24, 19), (27, 12), (20, 27), (28, 0), (23, 0), (21, 18), (27, 19), (25, 12), (23, 3), (24, 30),
        (26, 22), (29, 12), (29, 11), (21, 11), (29, 0), (26, 31), (23, 4), (27, 31), (33, 12), (31, 4), (28, 22),
        (34, 5), (29, 26), (33, 17), (26, 19), (24, 27), (21, 25), (30, 8), (32, 0), (30, 5), (21, 30), (30, 17),
        (29, 24), (19, 21), (35, 23), (30, 6), (26, 18), (22, 1), (27, 29), (28, 28), (29, 1), (34, 0), (32, 7),
        (26, 24), (31, 29), (34, 21), (29, 10), (33, 30), (33, 6), (35, 2), (20, 2), (33, 18), (25, 13), (35, 19),
        (27, 17), (28, 3), (33, 16), (19, 23), (21, 27), (23, 13), (22, 3), (21, 19), (28, 6), (29, 8), (23, 12),
        (25, 26), (26, 3), (34, 17), (21, 8), (29, 17), (31, 26), (24, 17), (29, 3), (25, 10), (28, 21), (29, 13),
        (34, 12), (32, 17), (24, 8), (21, 24), (24, 26), (34, 7), (25, 6), (24, 16), (19, 7), (31, 28), (17, 7),
        (20, 3), (34, 31), (20, 8), (22, 14), (27, 4), (21, 2), (28, 5)],
    # Appendix F.8. The top 256 heads of the language-head search on the colored-shapes task. The
    # first 16 are HEADS["language"].
    "language_ranking": [
        (24, 27), (32, 17), (22, 31), (20, 27), (30, 20), (31, 2), (35, 25), (35, 26), (31, 24), (31, 26), (34, 15),
        (22, 19), (22, 8), (24, 25), (35, 5), (19, 21), (32, 31), (32, 16), (22, 14), (35, 22), (35, 2), (27, 16),
        (22, 10), (18, 17), (34, 23), (33, 7), (19, 6), (23, 2), (12, 15), (34, 28), (19, 4), (32, 30), (31, 30),
        (34, 18), (25, 12), (20, 3), (31, 13), (35, 8), (25, 6), (19, 31), (26, 20), (35, 19), (18, 20), (24, 28),
        (24, 26), (32, 2), (21, 6), (17, 10), (19, 5), (17, 25), (2, 3), (31, 6), (14, 30), (32, 21), (23, 30),
        (35, 9), (34, 21), (0, 0), (25, 17), (8, 25), (23, 0), (34, 17), (12, 30), (18, 24), (6, 6), (35, 23),
        (33, 30), (23, 8), (23, 21), (26, 0), (22, 15), (20, 8), (26, 27), (23, 16), (17, 7), (34, 12), (9, 14),
        (21, 26), (27, 15), (17, 4), (16, 31), (22, 29), (10, 10), (10, 18), (9, 13), (22, 2), (28, 16), (16, 6),
        (17, 11), (24, 5), (19, 29), (17, 17), (10, 24), (21, 31), (20, 14), (31, 18), (30, 7), (19, 9), (21, 14),
        (19, 0), (33, 6), (5, 18), (24, 17), (26, 17), (21, 9), (21, 2), (22, 28), (14, 7), (30, 22), (14, 2),
        (23, 24), (29, 27), (21, 17), (12, 31), (22, 11), (18, 31), (25, 7), (21, 4), (30, 9), (24, 7), (14, 24),
        (16, 19), (24, 8), (33, 31), (17, 15), (29, 4), (7, 10), (25, 30), (18, 25), (12, 11), (10, 22), (13, 26),
        (14, 21), (18, 5), (20, 15), (17, 16), (31, 11), (15, 5), (10, 17), (4, 17), (17, 6), (9, 22), (3, 4),
        (19, 3), (33, 21), (32, 24), (35, 13), (23, 3), (3, 21), (23, 10), (31, 29), (5, 2), (25, 18), (20, 2),
        (18, 4), (0, 26), (5, 20), (34, 0), (14, 3), (26, 18), (16, 8), (12, 7), (8, 9), (12, 20), (18, 14),
        (10, 29), (29, 9), (7, 2), (4, 24), (5, 31), (3, 18), (22, 3), (26, 29), (33, 20), (21, 12), (26, 11),
        (11, 25), (5, 0), (23, 28), (16, 12), (6, 1), (6, 9), (14, 0), (11, 5), (28, 28), (7, 25), (25, 23),
        (18, 1), (20, 29), (32, 9), (16, 14), (25, 13), (6, 23), (2, 0), (12, 5), (33, 19), (12, 27), (21, 13),
        (19, 30), (20, 31), (13, 4), (27, 8), (19, 26), (25, 28), (0, 24), (4, 0), (17, 21), (28, 22), (23, 29),
        (3, 24), (11, 0), (7, 27), (19, 14), (23, 19), (15, 8), (19, 15), (15, 10), (15, 20), (15, 16), (9, 12),
        (4, 30), (14, 13), (20, 21), (5, 1), (13, 23), (33, 28), (10, 23), (17, 12), (11, 7), (13, 17), (10, 13),
        (15, 27), (14, 20), (14, 23), (16, 26), (3, 29), (34, 31), (29, 7), (28, 4), (33, 0), (2, 24), (24, 19),
        (5, 17), (13, 3), (8, 30), (35, 0), (15, 15), (8, 1), (13, 31), (28, 21), (19, 20), (1, 6), (19, 17),
        (27, 1), (12, 12), (25, 5)],
    # The 16 lowest-ranked heads of the same search (a control in Appendix F.6).
    "language_lowest": [
        (24, 29), (21, 15), (29, 6), (14, 26), (10, 31), (17, 24), (27, 18), (16, 13), (3, 11), (34, 29), (35, 27),
        (17, 19), (20, 9), (23, 15), (24, 31), (22, 16)],
    # Appendix F.6. The 16 language heads found by the same search on the object task (an earlier
    # run on 12 images). 9 of them are in HEADS["language"].
    "language_objects": [
        (24, 27), (32, 17), (20, 27), (35, 26), (22, 31), (35, 25), (30, 20), (22, 10), (19, 4), (22, 19), (22, 14),
        (34, 12), (31, 24), (35, 2), (34, 28), (22, 11)],
    # The top 256 heads of the search on the object task.
    "language_objects_ranking": [
        (24, 27), (32, 17), (20, 27), (35, 26), (22, 31), (35, 25), (30, 20), (22, 10), (19, 4), (22, 19), (22, 14),
        (34, 12), (31, 24), (35, 2), (34, 28), (22, 11), (34, 8), (24, 25), (31, 2), (31, 30), (31, 26), (34, 15),
        (22, 0), (33, 7), (23, 8), (30, 7), (19, 21), (28, 7), (33, 10), (24, 29), (23, 30), (22, 1), (32, 31),
        (23, 26), (17, 18), (27, 12), (19, 5), (12, 16), (25, 14), (10, 6), (21, 31), (25, 15), (10, 22), (28, 0),
        (21, 6), (24, 26), (35, 5), (23, 0), (26, 3), (21, 30), (28, 23), (32, 30), (25, 12), (27, 20), (19, 29),
        (26, 20), (25, 16), (29, 27), (33, 30), (25, 5), (31, 28), (24, 19), (27, 18), (12, 25), (21, 2), (19, 0),
        (22, 29), (19, 23), (33, 9), (23, 3), (9, 2), (25, 6), (23, 4), (35, 13), (28, 28), (11, 0), (31, 18),
        (34, 17), (16, 2), (23, 17), (31, 6), (29, 10), (22, 3), (9, 27), (22, 28), (29, 0), (23, 15), (33, 12),
        (19, 1), (8, 3), (17, 16), (24, 23), (26, 31), (20, 17), (21, 18), (35, 19), (24, 30), (26, 18), (23, 10),
        (3, 18), (2, 3), (16, 6), (12, 15), (13, 27), (20, 26), (28, 2), (12, 29), (24, 31), (16, 3), (19, 14),
        (18, 0), (16, 26), (20, 2), (19, 16), (19, 31), (18, 19), (22, 18), (25, 7), (27, 7), (29, 11), (21, 25),
        (22, 6), (25, 18), (33, 16), (27, 29), (16, 31), (22, 8), (14, 26), (33, 28), (34, 23), (31, 3), (29, 7),
        (17, 17), (17, 14), (34, 16), (23, 29), (11, 31), (35, 24), (35, 12), (14, 12), (20, 6), (16, 8), (29, 6),
        (0, 0), (11, 13), (35, 23), (33, 31), (20, 11), (33, 13), (17, 27), (20, 1), (31, 31), (16, 12), (26, 19),
        (28, 3), (7, 8), (25, 30), (20, 18), (32, 8), (19, 15), (10, 31), (8, 9), (28, 6), (12, 18), (32, 9),
        (10, 29), (35, 9), (11, 2), (4, 22), (23, 18), (31, 4), (10, 20), (32, 16), (9, 20), (23, 11), (21, 11),
        (21, 15), (21, 17), (3, 11), (23, 13), (28, 21), (15, 9), (18, 9), (10, 24), (22, 30), (34, 13), (25, 13),
        (32, 14), (22, 15), (20, 12), (18, 30), (17, 2), (12, 0), (18, 6), (19, 19), (31, 13), (17, 5), (4, 18),
        (26, 27), (18, 14), (33, 14), (29, 1), (34, 18), (25, 11), (16, 28), (28, 11), (13, 22), (21, 27), (5, 31),
        (7, 25), (18, 29), (23, 7), (3, 16), (27, 27), (12, 19), (2, 27), (14, 1), (14, 13), (30, 6), (15, 20),
        (32, 4), (15, 3), (31, 11), (27, 4), (20, 14), (27, 11), (9, 4), (16, 14), (12, 14), (13, 19), (23, 2),
        (8, 27), (31, 29), (14, 25), (34, 11), (15, 31), (31, 5), (4, 20), (19, 17), (16, 15), (29, 13), (21, 19),
        (20, 5), (7, 12), (10, 11), (26, 17), (18, 20), (12, 4), (33, 19), (30, 25), (8, 17), (18, 13), (14, 4),
        (31, 14), (18, 4), (23, 25)],
    # The 16 lowest-ranked heads of the search on the object task.
    "language_objects_lowest": [
        (17, 19), (32, 3), (24, 28), (16, 4), (35, 8), (34, 7), (12, 31), (30, 2), (14, 30), (9, 22), (24, 24),
        (22, 9), (34, 29), (34, 30), (22, 16), (35, 27)],
})


# ==================================================================================================
# Appendix G: other model sizes and families
# ==================================================================================================
HEADS.update({
    # Qwen3-VL-2B-Instruct: 28 layers, 16 heads per layer. Searched on 100 pairs, K chosen on 50.
    # Binding heads (head outputs at the shape token), 8 heads.
    "binding_2b": [(13, 6), (11, 13), (12, 5), (13, 14), (11, 11), (14, 15), (11, 2), (14, 14)],
    # Query heads: no K reached 80% on the 2B, so the rule took the best one, 64.
    "query_2b": [(11, 13), (13, 6), (13, 14), (11, 11), (14, 15), (11, 2), (12, 8), (14, 0), (11, 3), (14, 1), (11, 10),
                 (14, 14), (14, 6), (14, 10), (14, 7), (12, 0), (0, 12), (11, 1), (7, 13), (15, 11), (14, 11), (10, 11),
                 (9, 12), (12, 9), (10, 2), (6, 7), (16, 12), (12, 5), (12, 14), (13, 7), (9, 13), (9, 9), (15, 15),
                 (13, 4), (0, 1), (12, 12), (0, 3), (15, 1), (14, 13), (8, 10), (13, 0), (0, 7), (5, 15), (13, 8),
                 (16, 8), (14, 2), (15, 14), (12, 2), (15, 4), (8, 11), (4, 12), (13, 3), (12, 11), (12, 4), (12, 3),
                 (17, 12), (3, 6), (5, 7), (3, 5), (14, 5), (11, 14), (8, 15), (10, 5), (5, 5)],
    "key_2b": [(11, 13), (13, 6), (11, 11), (13, 14), (11, 2), (12, 8), (11, 3), (14, 15)],
    "relay_2b": [(16, 4), (18, 15), (25, 8), (17, 1), (23, 6), (17, 12), (19, 6), (27, 15), (17, 9), (18, 7), (18, 14),
                 (15, 14), (15, 4), (16, 5), (17, 2), (16, 8)],

    # Qwen3-VL-4B-Instruct: 36 layers, 32 heads per layer. Searched on 100 pairs, K chosen on 50.
    "binding_4b": [(17, 4), (18, 15), (15, 9), (19, 27), (15, 11), (19, 24), (15, 2), (21, 11)],
    "query_4b": [(19, 27), (17, 4), (15, 9), (15, 11), (18, 15), (19, 24), (15, 2), (15, 10)],
    "key_4b": [(19, 27), (15, 9), (15, 11), (15, 2), (19, 24), (18, 15), (17, 4), (17, 27)],
    "relay_4b": [(22, 10), (23, 26), (22, 11), (25, 15), (34, 28), (35, 25), (23, 0), (19, 21), (22, 1), (28, 22), (30, 6),
                 (23, 3), (24, 31), (21, 30), (22, 0), (21, 24)],

    # Qwen3-VL-32B-Instruct: 64 layers, 64 heads per layer. Searched on 20 pairs, K chosen on 10.
    "binding_32b": [(45, 20), (45, 11), (46, 29), (42, 7), (45, 10), (45, 8), (45, 17), (46, 26), (42, 4), (48, 31),
                    (46, 27), (46, 31), (42, 31), (45, 13), (42, 1), (44, 21)],
    "query_32b": [(45, 11), (45, 20), (46, 29), (45, 10), (42, 7), (42, 4), (45, 8), (46, 26), (46, 27), (42, 31),
                  (46, 31), (45, 17), (40, 61), (45, 13), (45, 6), (42, 1)],
    "key_32b": [(45, 11), (46, 29), (45, 20), (45, 10), (42, 7), (45, 8), (42, 4), (46, 27), (46, 26), (42, 31), (42, 1),
                (45, 22), (45, 6), (45, 17), (45, 13), (46, 28)],
    "relay_32b": [(51, 20), (54, 59), (51, 16), (47, 61), (49, 30), (54, 62), (51, 19), (49, 38), (49, 39), (46, 4),
                  (51, 22), (62, 39), (54, 56), (63, 22), (63, 16), (49, 24)],
    # Gaze heads (queries patched at the colon token): K = 64 passed at 95% on the validation pairs.
    "gaze_32b": [(52, 36), (61, 12), (61, 9), (52, 33), (50, 61), (54, 22), (52, 35), (62, 62), (52, 51), (53, 35),
                 (61, 11), (60, 45), (52, 48), (59, 30), (50, 58), (62, 59), (50, 60), (56, 20), (54, 62), (48, 31),
                 (53, 19), (62, 37), (58, 59), (49, 46), (52, 39), (58, 11), (48, 28), (62, 56), (62, 39), (48, 50),
                 (58, 40), (57, 11), (58, 63), (50, 30), (56, 22), (58, 8), (47, 22), (48, 29), (53, 37), (45, 20),
                 (49, 24), (53, 33), (49, 63), (46, 29), (51, 17), (49, 30), (48, 25), (59, 25), (55, 39), (59, 5),
                 (51, 49), (50, 63), (54, 16), (52, 32), (49, 62), (54, 23), (58, 56), (46, 28), (50, 62), (45, 12),
                 (45, 18), (59, 26), (47, 54), (63, 22)],
    # The answer-token search on the 32B (00_find_answer_token_heads_32b.py): the best 256 heads of its
    # ranking on 100 search pairs, best first. The first 64 are the 32B's answer-token heads.
    "answer_token_ranking_32b": [
        (45, 11), (47, 53), (50, 59), (45, 13), (40, 61), (49, 46), (45, 15), (52, 34), (46, 35), (45, 14), (52, 36),
        (49, 63), (45, 6), (33, 5), (49, 24), (39, 38), (51, 0), (40, 58), (40, 59), (48, 55), (46, 62), (51, 19),
        (48, 46), (40, 57), (44, 15), (45, 8), (50, 61), (41, 41), (41, 44), (45, 10), (53, 38), (33, 6), (42, 4),
        (40, 62), (48, 28), (39, 39), (61, 15), (45, 20), (45, 7), (47, 54), (48, 2), (39, 35), (45, 33), (59, 30),
        (45, 17), (40, 63), (44, 37), (51, 49), (46, 39), (49, 29), (33, 2), (42, 1), (42, 2), (43, 24), (47, 48),
        (42, 46), (49, 26), (46, 26), (43, 34), (55, 55), (48, 30), (55, 39), (52, 37), (42, 43), (49, 34), (45, 41),
        (50, 36), (44, 36), (47, 52), (60, 45), (45, 62), (52, 38), (59, 5), (48, 5), (42, 7), (56, 18), (44, 35),
        (28, 35), (46, 55), (48, 58), (49, 22), (53, 20), (51, 30), (47, 50), (47, 47), (42, 44), (42, 30), (45, 56),
        (44, 44), (41, 38), (45, 53), (43, 52), (47, 9), (43, 37), (54, 26), (53, 19), (45, 32), (50, 25), (62, 32),
        (49, 62), (50, 10), (44, 25), (41, 7), (50, 29), (48, 49), (39, 12), (39, 36), (61, 63), (51, 42), (42, 6),
        (43, 58), (47, 18), (55, 48), (55, 53), (55, 9), (42, 48), (44, 10), (54, 63), (45, 30), (45, 44), (43, 55),
        (61, 13), (51, 39), (60, 40), (46, 53), (48, 54), (51, 3), (55, 12), (42, 52), (42, 28), (55, 37), (46, 50),
        (49, 9), (45, 26), (48, 14), (33, 0), (51, 1), (41, 56), (45, 5), (41, 2), (48, 63), (49, 37), (41, 3),
        (47, 42), (51, 24), (56, 39), (47, 16), (41, 55), (46, 28), (46, 31), (57, 14), (56, 21), (43, 59), (44, 51),
        (52, 7), (44, 34), (41, 14), (51, 29), (43, 33), (51, 4), (39, 4), (42, 55), (46, 41), (60, 13), (54, 17),
        (48, 34), (58, 23), (46, 4), (48, 44), (46, 48), (44, 47), (47, 25), (39, 9), (48, 12), (48, 21), (44, 8),
        (36, 41), (43, 45), (40, 56), (45, 48), (43, 22), (48, 17), (51, 58), (46, 58), (58, 56), (60, 12), (45, 61),
        (47, 33), (48, 25), (45, 35), (56, 60), (40, 38), (61, 48), (50, 1), (48, 27), (62, 35), (46, 19), (45, 40),
        (43, 63), (44, 50), (40, 35), (43, 10), (45, 25), (57, 25), (45, 12), (54, 27), (60, 46), (39, 43), (50, 33),
        (43, 20), (51, 8), (60, 28), (53, 47), (54, 57), (61, 40), (46, 44), (42, 42), (45, 42), (42, 22), (54, 34),
        (46, 16), (47, 58), (59, 60), (61, 62), (41, 61), (56, 61), (49, 50), (39, 37), (57, 23), (49, 31), (53, 10),
        (56, 36), (47, 31), (43, 23), (52, 54), (54, 22), (57, 43), (45, 59), (55, 52), (43, 32), (45, 36), (45, 50),
        (46, 38), (43, 43), (47, 8), (53, 22), (51, 55), (57, 52), (49, 42), (54, 48), (56, 57), (45, 45), (58, 6),
        (57, 18), (54, 32), (48, 16)],
})

MODEL_SIZES = {"Qwen/Qwen3-VL-2B-Instruct": "2b", "Qwen/Qwen3-VL-4B-Instruct": "4b",
               "Qwen/Qwen3-VL-8B-Instruct": "8b", "Qwen/Qwen3-VL-32B-Instruct": "32b"}


def heads_of_size(model_id, name):
    """The head set `name` of one Qwen3-VL size, as (layer, head) pairs.

    name   binding, query, key, relay, gaze, language, answer_token, or one of these unions:
           query_and_key          the query heads and the key heads
           binding_query_key      the binding, query and key heads
           blocked_at_shape_token the heads the paper blocks from the image at the shape token: on the
                                  8B the binding, query and key heads, on the other sizes the query and
                                  key heads (the set the paper's runs used there)
           rebind                 the heads that read the source's image keys at an answer token in the
                                  rebind: on the 8B the binding heads, on the other sizes the binding,
                                  query and key heads (the set the paper's runs used there)
    The 8B uses the sets of the main paper (HEADS without a suffix)."""
    assert model_id in MODEL_SIZES, f"{model_id} is not a Qwen3-VL size; choose from {list(MODEL_SIZES)}"
    size = MODEL_SIZES[model_id]
    def union(*names):
        return sorted({head for n in names for head in heads_of_size(model_id, n)})

    if name == "query_and_key":
        return union("query", "key")
    if name == "binding_query_key":
        return union("binding", "query", "key")
    if name == "blocked_at_shape_token":
        return union("binding", "query", "key") if size == "8b" else union("query", "key")
    if name == "rebind":
        return heads_of_size(model_id, "binding") if size == "8b" else union("binding", "query", "key")
    if size == "8b":
        return HEADS[name]
    if name == "answer_token" and size == "32b":
        return HEADS["answer_token_ranking_32b"][:64]
    key = f"{name}_{size}"
    if key not in HEADS:
        raise KeyError(f"no {name} heads for {model_id} in utils/heads.py. "
                       "Find them with generalization/main_experiments/00_find_heads_at_each_size.py.")
    return HEADS[key]


def random_heads_like(model, heads, seed, exclude=()):
    """`random_like` for any model size: one random head per given head, from the same layer."""
    from utils.model import n_heads, n_layers
    return random_like(heads, seed, n_layers=n_layers(model), n_heads=n_heads(model), exclude=exclude)


def sweep_k_letter(model, pairs, ranking, build, letter, sizes):
    """Step 2 of a head search on any size: patch the top K heads of a ranking together, for every K
    in `sizes`, on both directions of each pair. Returns {K: rate of the outcome letter}.

    The 8B search (`sweep_k`) also reads the published Gaze Heads, which exist only on the 8B. The
    paper's runs on the other sizes chose K on the letter rate alone, as this function does."""
    from utils.dataset import both_directions
    from utils.patching import run_patches
    hits, cases = {k: 0 for k in sizes}, 0
    for pair in pairs:
        for source, target, letters in both_directions(model, pair):
            answers = run_patches(model, target, [build(source, target, ranking[:k]) for k in sizes])["answers"]
            for k, answer in zip(sizes, answers):
                hits[k] += answer == " " + letters[letter]
            cases += 1
    return {k: hits[k] / max(cases, 1) for k in sizes}


# ==================================================================================================
# Appendix G: other layouts, prompts, and natural images
# ==================================================================================================
HEADS.update({
    # Appendix G.3.1. The heads selected on the first, comic version of the one-answer task, in the
    # order of their search ranking: 8 query heads and 8 key heads at the shape token, 32 output heads
    # at the shape token, and 16 relay heads at the colon token.
    "comic_query": [(15, 2), (15, 9), (15, 11), (17, 4), (18, 15), (19, 24), (19, 27), (21, 11)],
    "comic_key": [(15, 2), (15, 9), (15, 11), (17, 4), (17, 21), (18, 15), (19, 24), (19, 27)],
    "comic_output": [(19, 27), (18, 15), (15, 9), (19, 24), (15, 11), (17, 4), (21, 11), (17, 7), (15, 2), (21, 18),
                     (20, 29), (18, 11), (18, 8), (17, 15), (16, 27), (20, 15), (21, 8), (14, 30), (19, 21), (21, 19),
                     (19, 14), (18, 14), (19, 3), (21, 30), (20, 2), (18, 12), (18, 21), (20, 16), (6, 15), (18, 16),
                     (12, 24), (20, 31)],
    "comic_relay": [(23, 26), (34, 28), (22, 10), (35, 25), (22, 11), (25, 15), (30, 6), (23, 0), (24, 31), (21, 19),
                    (26, 26), (29, 6), (34, 30), (20, 2), (25, 12), (20, 3)],
    # The 16 language heads selected with the colored shapes drawn over comic strips.
    "comic_language": [(24, 27), (32, 17), (22, 31), (31, 2), (30, 20), (35, 25), (20, 27), (35, 26), (31, 24), (31, 26),
                       (34, 15), (32, 16), (24, 25), (22, 19), (19, 6), (35, 5)],
    # Appendix G.3.5. The 16 heads that read " List them from right to left." (00_find_instruction_heads.py).
    "instruction": [(18, 11), (18, 17), (18, 19), (19, 0), (19, 2), (19, 12), (19, 23), (20, 0), (20, 1), (20, 28),
                    (21, 16), (21, 17), (21, 26), (22, 2), (22, 9), (23, 24)],
    # Appendix G.3.7. The 16 answer-token heads found with no image, when the panels are described in
    # words (00_find_text_only_answer_token_heads.py), in the order of their search ranking.
    "text_only_answer_token": [(19, 24), (19, 27), (20, 31), (18, 15), (24, 29), (15, 9), (21, 19), (20, 30), (22, 30),
                               (21, 11), (15, 11), (24, 31), (17, 4), (19, 13), (20, 28), (18, 10)],
})
