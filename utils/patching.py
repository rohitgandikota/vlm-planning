"""Run the model, record its internal states, and patch them.

The three things every experiment in the paper does:

    source = run_model(model, source_image, source_question)      # 1. run and record
    target = run_model(model, target_image, target_question)

    patch = Patch()                                               # 2. describe the edit
    patch.head_output(layer=17, head=4, token=target.token("shape"),
                      vector=source.head_output(17, 4, source.token("shape")))

    answer = run_patch(model, target, patch)                      # 3. rerun the target, read its next token

What can be patched, at any text token after the image:
    residual      the residual vector entering a layer (layer = number of layers means after the last one)
    mlp           the output of one MLP
    head_output   the output of one attention head (before the output projection)
    query         the query vector of one attention head
    keys          the keys one head sees at chosen positions (for example the image tokens)
    values        the values one head reads from chosen positions
    block         remove attention from chosen tokens to chosen positions

Positions are always indices into the full token sequence (image tokens included).

How it works. The image comes before the question in the prompt, so we run the model once through
the last image token, keep that part as a cache, and recompute only the text tokens after the image
for every patch. This gives the same logits as a full forward pass (`check_parity`) and is much faster.
The attention edits are made inside the model's own attention function, which is replaced below.
"""
import copy
from collections import defaultdict

import numpy as np
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from utils.model import n_heads, n_layers, panel_of_image_tokens, prepare

_ORIGINAL_ATTENTION = qwen.eager_attention_forward
_ACTIVE = {"edits": None}          # set while a patched run is in progress
_KINDS = ("query", "keys", "values", "head_output", "block")


def _index(positions):
    """Positions as a CPU tensor of integers, from a list, a range, or a tensor."""
    if torch.is_tensor(positions):
        return positions.detach().cpu().long().reshape(-1)
    return torch.as_tensor(list(positions), dtype=torch.long)


def _attention_with_edits(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """The model's attention, with the edits of the current run applied. Rows of the batch carry
    different edits, so every edit starts with its batch row."""
    state = _ACTIVE["edits"]
    layer = getattr(module, "layer_idx", None)
    if state is None or layer is None:                       # not one of our runs, or the vision encoder
        return _ORIGINAL_ATTENTION(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    state["calls"] += 1
    edits = state["attention"]
    keys = qwen.repeat_kv(key, module.num_key_value_groups)        # one key and value per query head
    values = qwen.repeat_kv(value, module.num_key_value_groups)
    if edits["query"].get(layer):
        query = query.clone()
        for row, head, token, vector in edits["query"][layer]:
            query[row, head, token] = vector
    scores = torch.matmul(query, keys.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    for row, head, token, positions, new_keys in edits["keys"].get(layer, ()):
        scores[row, head, token, positions] = (new_keys @ query[row, head, token]) * scaling
    lowest = torch.finfo(scores.dtype).min
    for row, heads, tokens, positions in edits["block"].get(layer, ()):
        scores[row, heads[:, None, None], tokens[None, :, None], positions[None, None, :]] = lowest
    weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(weights, values)
    for row, head, token, positions, new_values in edits["values"].get(layer, ()):
        out[row, head, token] += weights[row, head, token, positions] @ (new_values - values[row, head, positions])
    for row, head, token, vector in edits["head_output"].get(layer, ()):
        out[row, head, token] = vector
    if state["record"] is not None:                           # keep the states of batch row 0
        state["record"][layer] = {"query": query[0].clone(), "head_output": out[0].clone(),
                                  "keys": key[0].clone(), "values": value[0].clone()}
    if state["keep_attention"]:
        state["last_token_attention"][layer] = weights[:, :, -1, :].detach()
    return out.transpose(1, 2).contiguous(), weights


qwen.eager_attention_forward = _attention_with_edits


# ---------------------------------------------------------------------------------------------- Patch
class Patch:
    """A list of edits to apply in one run of the target. Build it with the methods below.

    `token` is the position of the text token being computed. `positions` are the positions a head
    attends to (image tokens or earlier text tokens). Vectors come from a recorded source run."""

    def __init__(self, name=""):
        self.name = name
        self.attention = []        # (kind, layer, payload)
        self.residuals = []        # (layer, token, vector)
        self.mlps = []             # (layer, token, vector)

    def residual(self, layer, token, vector):
        self.residuals.append((layer, token, vector))
        return self

    def mlp(self, layer, token, vector):
        self.mlps.append((layer, token, vector))
        return self

    def query(self, layer, head, token, vector):
        self.attention.append(("query", layer, (head, token, vector)))
        return self

    def head_output(self, layer, head, token, vector):
        self.attention.append(("head_output", layer, (head, token, vector)))
        return self

    def keys(self, layer, head, token, positions, keys):
        """The head scores `positions` with these keys instead of the target's own keys."""
        self.attention.append(("keys", layer, (head, token, _index(positions), keys)))
        return self

    def values(self, layer, head, token, positions, values):
        """The head reads these values from `positions`, with its attention weights unchanged."""
        self.attention.append(("values", layer, (head, token, _index(positions), values)))
        return self

    def block(self, heads, tokens, positions, layers=None):
        """Remove the attention from `tokens` to `positions`.

        heads: a list of (layer, head), or None for every head of every layer in `layers`."""
        tokens, positions = _index(tokens), _index(positions)
        if heads is None:
            for layer in layers:
                self.attention.append(("block", layer, (None, tokens, positions)))
            return self
        by_layer = defaultdict(list)
        for layer, head in heads:
            if layers is None or layer in layers:
                by_layer[layer].append(head)
        for layer, layer_heads in by_layer.items():
            self.attention.append(("block", layer, (torch.as_tensor(sorted(layer_heads)), tokens, positions)))
        return self

    def add(self, other):
        """Add all the edits of another patch to this one."""
        self.attention += other.attention
        self.residuals += other.residuals
        self.mlps += other.mlps
        return self


# ---------------------------------------------------------------------------------------------- Run
class Run:
    """One recorded run of the model on an image and a prompt.

    After `run_model`, a Run gives every state of the text tokens after the image:
        run.residual(layer, token)            run.mlp_output(layer, token)
        run.query(layer, head, token)         run.head_output(layer, head, token)
        run.keys(layer, head, positions)      run.values(layer, head, positions)
    and the model's own answer: run.answer (text), run.answer_id, run.logits.
    Named tokens are set with `run.name_token("shape", position)` and read with `run.token("shape")`."""

    def __init__(self, model, image, question, prefill="Answer:", regions=None, record=True):
        inputs, info = prepare(model, image, question, prefill)
        self.model, self.image, self.question, self.prefill = model, image, question, prefill
        self.info, self.ids = info, info["input_ids"]
        self.tokenizer = model.processor.tokenizer
        self.device = model.device
        self.regions = regions if regions is not None else getattr(image, "regions", None)
        self.image_positions = torch.as_tensor(info["image_positions"], device=self.device)
        self.first_text = info["image_positions"][-1] + 1            # the first token after the image
        self.last = len(self.ids) - 1                                 # the token whose next token we read
        self.names = {"last": self.last}
        self._kv_group = n_heads(model) // model.config.text_config.num_key_value_heads
        self._cache_prefix(inputs)
        self.states = None
        result = _recompute(model, self, [Patch()], record=record)
        self.logits = result["logits"][0]
        self.answer_id = int(self.logits.argmax())
        self.states = result.get("states")

    # --- tokens
    def _cache_prefix(self, inputs):
        """Run the model through the last image token once and keep the result."""
        model = self.model
        positions, _ = model.model.get_rope_index(
            input_ids=inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
            image_grid_thw=inputs["image_grid_thw"], attention_mask=inputs["attention_mask"])
        before = dict(inputs)
        for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
            before[name] = before[name][:, :self.first_text]
        before["position_ids"] = positions[..., :self.first_text]
        model.rope_deltas = None
        output = model(**before, use_cache=True, logits_to_keep=1)
        assert output.past_key_values.get_seq_length() == self.first_text
        self._cache = output.past_key_values
        self._text_ids = inputs["input_ids"][:, self.first_text:]
        self._mask = inputs["attention_mask"]
        self._positions = positions[..., self.first_text:]
        self._inputs = inputs

    def text_tokens(self):
        """The text tokens after the image as (position, text) pairs. Useful to pick a token to patch."""
        return [(self.first_text + i, self.tokenizer.decode([t])) for i, t in enumerate(self.ids[self.first_text:])]

    def find(self, text, which=-1):
        """The position of a token after the image by its text, for example find(" star"). `which`
        picks among several matches (-1 is the last one)."""
        matches = [p for p, t in self.text_tokens() if t == text]
        assert matches, f"token {text!r} not found; tokens are {[t for _, t in self.text_tokens()]}"
        return matches[which]

    def name_token(self, name, position):
        self.names[name] = position
        return self

    def token(self, name):
        """A position from a name set with `name_token`, or the position itself."""
        return self.names[name] if isinstance(name, str) else name

    def decode(self, token_id):
        return "<end>" if token_id == self.info["end_of_turn"] else self.tokenizer.decode([token_id])

    @property
    def answer(self):
        """The next token of the unpatched run, as text."""
        return self.decode(self.answer_id)

    # --- recorded states (positions are indices into the full sequence)
    def _t(self, token):
        index = self.token(token) - self.first_text
        assert index >= 0, "only text tokens after the image are recomputed and recorded"
        return index

    def residual(self, layer, token):
        return self.states["residual"][layer][self._t(token)]

    def mlp_output(self, layer, token):
        return self.states["mlp"][layer][self._t(token)]

    def query(self, layer, head, token):
        return self.states["attention"][layer]["query"][head, self._t(token)]

    def head_output(self, layer, head, token):
        return self.states["attention"][layer]["head_output"][head, self._t(token)]

    def keys(self, layer, head, positions):
        return self.states["attention"][layer]["keys"][head // self._kv_group][_index(positions).to(self.device)]

    def values(self, layer, head, positions):
        return self.states["attention"][layer]["values"][head // self._kv_group][_index(positions).to(self.device)]

    def attention(self, layer, head, token):
        """The attention weights of one head at one token over all earlier positions, from the
        recorded query and keys."""
        token = self.token(token)
        query = self.query(layer, head, token)
        keys = self.states["attention"][layer]["keys"][head // self._kv_group][: token + 1]
        return torch.softmax(keys @ query / query.shape[-1] ** 0.5, dim=0)

    # --- image regions
    def panel_of_image_tokens(self):
        """The panel index of every image token (-1 if in no panel), as a tensor."""
        return torch.as_tensor(panel_of_image_tokens(self.info, self.regions), device=self.device)

    def panel_positions(self, panel):
        """The positions of the image tokens that lie in one panel."""
        return self.image_positions[self.panel_of_image_tokens() == panel]

    def panel_attention(self, heads, token="last"):
        """Share of the image attention of `heads` at one token that falls on each panel."""
        panels = self.panel_of_image_tokens()
        mass = torch.zeros(len(self.regions), device=self.device)
        for layer, head in heads:
            weights = self.attention(layer, head, token)[self.image_positions]
            for panel in range(len(self.regions)):
                mass[panel] += weights[panels == panel].sum()
        return (mass / mass.sum()).tolist()


def run_model(model, image, question, prefill="Answer:", regions=None, record=True):
    """Run the model on one image and one question and record its states. See `Run`."""
    with torch.no_grad():
        return Run(model, image, question, prefill, regions, record)


# ---------------------------------------------------------------------------------------------- running patches
def _recompute(model, run, patches, record=False, keep_attention=False):
    """Recompute the text tokens of `run` once per patch, as one batch."""
    layers = model.model.language_model.layers
    modules = list(layers) + [model.model.language_model.norm]
    batch, shift, device = len(patches), run.first_text, run.device
    attention = {kind: defaultdict(list) for kind in _KINDS}
    residuals, mlps = defaultdict(list), defaultdict(list)
    all_heads = torch.arange(n_heads(model), device=device)
    for row, patch in enumerate(patches):
        for kind, layer, payload in patch.attention:
            if kind == "block":
                heads, tokens, positions = payload
                heads = all_heads if heads is None else heads.to(device)
                attention[kind][layer].append((row, heads, (tokens - shift).to(device), positions.to(device)))
            elif kind in ("keys", "values"):
                head, token, positions, vectors = payload
                attention[kind][layer].append((row, head, token - shift, positions.to(device), vectors))
            else:
                head, token, vector = payload
                attention[kind][layer].append((row, head, token - shift, vector))
        for layer, token, vector in patch.residuals:
            residuals[layer].append((row, token - shift, vector))
        for layer, token, vector in patch.mlps:
            mlps[layer].append((row, token - shift, vector))
    state = {"attention": attention, "calls": 0, "record": {} if record else None,
             "keep_attention": keep_attention, "last_token_attention": {}}
    captured = {"residual": [None] * len(modules), "mlp": [None] * len(layers)}
    hooks = []
    for index, module in enumerate(modules):
        def before(_module, args, index=index):
            hidden = args[0]
            if residuals.get(index):
                hidden = hidden.clone()
                for row, token, vector in residuals[index]:
                    hidden[row, token] = vector
            if record:
                captured["residual"][index] = hidden[0].detach().clone()
            return (hidden,) + args[1:]
        hooks.append(module.register_forward_pre_hook(before))
    for index, block in enumerate(layers):
        def after(_module, _args, output, index=index):
            if mlps.get(index):
                output = output.clone()
                for row, token, vector in mlps[index]:
                    output[row, token] = vector
            if record:
                captured["mlp"][index] = output[0].detach().clone()
            return output
        hooks.append(block.mlp.register_forward_hook(after))
    cache = copy.deepcopy(run._cache)
    if batch > 1:
        cache.batch_repeat_interleave(batch)
    _ACTIVE["edits"] = state
    try:
        model.rope_deltas = None
        output = model(input_ids=run._text_ids.repeat(batch, 1), attention_mask=run._mask.repeat(batch, 1),
                       position_ids=run._positions.repeat(1, batch, 1), past_key_values=cache,
                       use_cache=True, logits_to_keep=1, return_dict=True)
    finally:
        _ACTIVE["edits"] = None
        for hook in hooks:
            hook.remove()
    assert state["calls"] == len(layers), "the attention edits did not run in every layer"
    result = {"logits": output.logits[:, -1].detach().float()}
    if keep_attention:
        result["attention"] = state["last_token_attention"]
    if record:
        result["states"] = {"attention": state["record"], "residual": torch.stack(captured["residual"]),
                            "mlp": torch.stack(captured["mlp"])}
    return result


def run_patches(model, target, patches, max_batch=16, keep_attention=False):
    """Rerun the target once per patch. Returns a dict with
        "logits"     next-token logits, one row per patch (on the CPU)
        "ids"        the most likely next token id per patch
        "answers"    the same as text
        "attention"  only with keep_attention=True: per layer, the last token's attention weights
                     with shape (patches, heads, positions)"""
    logits, attention = [], defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(patches), max_batch):
            result = _recompute(model, target, patches[start:start + max_batch], keep_attention=keep_attention)
            logits.append(result["logits"].cpu())
            for layer, weights in result.get("attention", {}).items():
                attention[layer].append(weights)
    logits = torch.cat(logits)
    ids = logits.argmax(dim=-1).tolist()
    out = {"logits": logits, "ids": ids, "answers": [target.decode(i) for i in ids]}
    if keep_attention:
        out["attention"] = {layer: torch.cat(parts) for layer, parts in attention.items()}
    return out


def run_patch(model, target, patch):
    """Rerun the target with one patch and return its next token as text."""
    return run_patches(model, target, [patch])["answers"][0]


def panel_attention(attention, heads, run):
    """From `run_patches(..., keep_attention=True)["attention"]`: for each patch, the share of the
    image attention of `heads` at the last token that falls on each panel. Returns a tensor
    (patches, panels)."""
    panels = run.panel_of_image_tokens()
    rows = next(iter(attention.values())).shape[0]
    mass = torch.zeros(rows, len(run.regions))
    for layer, head in heads:
        weights = attention[layer][:, head][:, run.image_positions].float().cpu()
        for panel in range(len(run.regions)):
            mass[:, panel] += weights[:, (panels == panel).cpu()].sum(dim=1)
    return mass / mass.sum(dim=1, keepdim=True)


def check_parity(model, run, tolerance=0.002):
    """Check that recomputing only the text tokens gives the same logits as a full forward pass."""
    with torch.no_grad():
        model.rope_deltas = None
        full = model(**run._inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    error = float((full - run.logits.to(full.device)).abs().max())
    assert error < tolerance and int(full.argmax()) == run.answer_id, f"recompute differs from the full pass by {error}"
    return error


# ---------------------------------------------------------------------------------------------- one-line patch
COMPONENTS = ("residual", "mlp", "head_output", "query", "image_keys", "image_values", "keys", "values")


def make_patch(source, target, component, token, source_token=None, layer=None, source_layer=None,
               heads=None, source_heads=None, positions=None, source_positions=None):
    """Build a patch that copies one component from a source run into a target run.

    component      one of COMPONENTS
    token          the target token that is patched (a position or a name such as "shape", "colon")
    source_token   the source token the state is taken from (default: the same name or position)
    layer          for "residual" and "mlp": the target layer. source_layer defaults to the same layer
    heads          for the attention components: a list of (layer, head) patched in the target
    source_heads   the heads the states are taken from in the source (default: the same heads)
    positions      for "keys" and "values": the positions the heads read from in the target
    source_positions  the positions the keys or values are taken from in the source (default: the same)

    "image_keys" and "image_values" are "keys" and "values" at all image tokens."""
    assert component in COMPONENTS, component
    patch = Patch(f"{component} at {token}")
    t = target.token(token)
    s = source.token(token if source_token is None else source_token)
    if component in ("residual", "mlp"):
        from_layer = layer if source_layer is None else source_layer
        if component == "residual":
            return patch.residual(layer, t, source.residual(from_layer, s))
        return patch.mlp(layer, t, source.mlp_output(from_layer, s))
    if component in ("image_keys", "image_values"):
        positions, source_positions = target.image_positions, source.image_positions
        component = component[len("image_"):]
    if component in ("keys", "values"):
        source_positions = positions if source_positions is None else source_positions
    for (layer_, head), (from_layer, from_head) in zip(heads, heads if source_heads is None else source_heads):
        if component == "query":
            patch.query(layer_, head, t, source.query(from_layer, from_head, s))
        elif component == "head_output":
            patch.head_output(layer_, head, t, source.head_output(from_layer, from_head, s))
        elif component == "keys":
            patch.keys(layer_, head, t, positions, source.keys(from_layer, from_head, source_positions))
        else:
            patch.values(layer_, head, t, positions, source.values(from_layer, from_head, source_positions))
    return patch


# ---------------------------------------------------------------------------------------------- recording a patched run
def record_patch(model, target, patch):
    """Rerun the target with a patch and record the states of that patched run.

    Returns a Run like `target` whose recorded states and answer are those of the patched run, so
    it can be used as the source of another patch (for example: which values does the colon token
    read after the binding heads were patched at the shape token?)."""
    with torch.no_grad():
        result = _recompute(model, target, [patch], record=True)
    patched = copy.copy(target)                       # same image, prompt and cache; new states
    patched.states = result["states"]
    patched.logits = result["logits"][0]
    patched.answer_id = int(patched.logits.argmax())
    return patched


# ---------------------------------------------------------------------------------------------- what heads write, and linear probes
def heads_contribution(model, run, heads, token):
    """What a set of heads writes into the residual stream at one token: every head's output passed
    through its slice of the layer's output projection, summed over the heads. Returns one vector."""
    total = 0
    for layer, head in heads:
        projection = model.model.language_model.layers[layer].self_attn.o_proj.weight
        output = run.head_output(layer, head, token)
        total = total + projection[:, head * len(output):(head + 1) * len(output)] @ output
    return total


def fit_linear_probe(vectors, targets, ridge=50.0):
    """Fit a linear probe with ridge regression. Returns a function that predicts targets for new vectors.

    vectors   (cases, dimensions), for example residual vectors
    targets   (cases, outputs), for example one 0/1 value per panel
    The vectors are standardized first. There are far fewer cases than dimensions, so we solve the
    equivalent small system with one row per case."""
    vectors, targets = np.asarray(vectors, np.float64), np.asarray(targets, np.float64)
    mean, scale, offset = vectors.mean(0), vectors.std(0) + 1e-6, targets.mean(0)
    z = (vectors - mean) / scale
    weights = z.T @ np.linalg.solve(z @ z.T + ridge * np.eye(len(z)), targets - offset)
    return lambda new: ((np.asarray(new, np.float64) - mean) / scale) @ weights + offset


# ---------------------------------------------------------------------------------------------- writing several tokens, and the LogitLens
def add_token(run, token_id, record=True):
    """A copy of a run with one more token at the end of its prompt. The image part is reused.
    With record=True the longer prompt is run and its states are recorded, as in `run_model`."""
    longer = copy.copy(run)
    longer.ids = run.ids + [token_id]
    longer.last = run.last + 1
    longer.names = {**run.names, "last": longer.last}
    longer._text_ids = torch.cat([run._text_ids, torch.tensor([[token_id]], device=run.device)], dim=1)
    longer._mask = torch.cat([run._mask, torch.ones_like(run._mask[:, :1])], dim=1)
    longer._positions = torch.cat([run._positions, run._positions[..., -1:] + 1], dim=-1)
    if record:
        with torch.no_grad():
            result = _recompute(run.model, longer, [Patch()], record=True)
        longer.logits, longer.states = result["logits"][0], result["states"]
        longer.answer_id = int(longer.logits.argmax())
    return longer


def write_tokens(model, target, source=None, heads=(), every_token=True, max_tokens=8):
    """Let the target write up to `max_tokens` tokens, one at a time, and return them as text.

    With a source, the outputs of `heads` are copied from the source into the target while it
    writes: at the last prompt token and, with every_token=True, at every written token too. The
    source reads the tokens the target writes, so the two runs stay in step. With no source this is
    plain greedy writing. Writing stops at the end of the turn."""
    tokenizer = target.tokenizer
    stop = {target.info["end_of_turn"], tokenizer.eos_token_id}
    first_target, first_source = target.last, source.last if source is not None else None
    written = []
    for step in range(max_tokens):
        patch = Patch()
        for back in range(step + 1 if every_token else 1):        # every token patched so far stays patched
            for layer, head in heads:
                patch.head_output(layer, head, first_target + back, source.head_output(layer, head, first_source + back))
        token = run_patches(model, target, [patch])["ids"][0]
        written.append(token)
        if token in stop or step == max_tokens - 1:
            break
        target = add_token(target, token, record=False)
        if source is not None and every_token:
            source = add_token(source, token)
    return tokenizer.decode([t for t in written if t not in stop])


def written_by_heads(model, run, heads, token="last"):
    """What a set of heads adds to the residual stream at one token: the sum of their outputs after
    the output projection of their layers. Returns one vector of the model's hidden size."""
    total = 0
    for layer, head in heads:
        out = run.head_output(layer, head, token)
        weight = model.model.language_model.layers[layer].self_attn.o_proj.weight
        total = total + weight[:, head * len(out):(head + 1) * len(out)].float() @ out.float()
    return total


def top_tokens(model, vector, k=5):
    """LogitLens: the k output tokens that a residual vector points to most strongly, as text."""
    scores = model.lm_head.weight.float() @ vector.to(model.lm_head.weight.device)
    return [model.processor.tokenizer.decode([int(t)]) for t in scores.topk(k).indices]


# ==================================================================================================
# Appendix E
# ==================================================================================================
def attention_share(run, heads, token):
    """For each head, the share of its whole attention at one token that falls on each panel.
    Returns a tensor (heads, panels). Unlike `Run.panel_attention`, the shares are not rescaled to
    the image: the rest of the attention goes to the text tokens."""
    panels = run.panel_of_image_tokens()
    shares = torch.zeros(len(heads), len(run.regions))
    for i, (layer, head) in enumerate(heads):
        weights = run.attention(layer, head, token)[run.image_positions].float().cpu()
        for panel in range(len(run.regions)):
            shares[i, panel] = weights[(panels == panel).cpu()].sum()
    return shares


def report_mean(label, values, pairs, paper=None, totals=None):
    """Print the mean of one number per case, in percent, with its 95% bootstrap interval over pairs.
    `values` are fractions (0 to 1); `pairs` names the pair of each case. With `totals`, print the
    ratio sum(values) / sum(totals) instead (for example attention mass on a panel over the mass on
    all panels). Returns the value."""
    from utils.experiment import interval
    totals = [1.0] * len(values) if totals is None else totals
    mean = 100 * sum(values) / max(sum(totals), 1e-12)
    low, high = interval(values, totals, pairs) if values else (float("nan"), float("nan"))
    text = f"{label:45s} {mean:5.1f}   [{low:.1f}, {high:.1f}]  ({len(values)} cases)"
    print(text + (f"   paper: {paper}" if paper is not None else ""), flush=True)
    return mean


# ==================================================================================================
# Appendix F
# ==================================================================================================
def write_with_patch(model, target, edit, every_token=False, max_tokens=8):
    """Let the target write up to `max_tokens` tokens, greedily, with an edit kept on. Returns the text.

    `edit(patch, token)` adds the edits for one token position to a Patch. It is called for the last
    prompt token, and with every_token=True also for every written token. An edit at the last prompt
    token stays on while the target writes, because every written token reads that patched token.
    Writing stops at the end of the turn."""
    tokenizer = target.tokenizer
    stop = {target.info["end_of_turn"], tokenizer.eos_token_id}
    first = target.last
    written = []
    for step in range(max_tokens):
        patch = Patch()
        for token in range(first, first + (step + 1 if every_token else 1)):
            edit(patch, token)
        token_id = run_patches(model, target, [patch])["ids"][0]
        written.append(token_id)
        if token_id in stop or step == max_tokens - 1:
            break
        target = add_token(target, token_id, record=False)
    return tokenizer.decode([t for t in written if t not in stop])


def set_head_outputs(heads, vectors):
    """An edit for `write_with_patch`: the outputs of `heads` set to fixed vectors at a token.
    `vectors` maps (layer, head) to a vector, for example a zero vector or an average output."""
    def edit(patch, token):
        for layer, head in heads:
            patch.head_output(layer, head, token, vectors[(layer, head)])
    return edit


# ==================================================================================================
# Appendix G: other model sizes and families
# ==================================================================================================
def write_with_fixed_patch(model, run, patch, max_tokens=8):
    """Let a Qwen3-VL run write up to `max_tokens` tokens, one at a time, with `patch` applied at
    every step, and return the text. The patch edits prompt tokens, which every step recomputes, so
    the written tokens always see the patched prompt. Writing stops at the end of the turn."""
    stop = {run.info["end_of_turn"], run.tokenizer.eos_token_id}
    written = []
    for _ in range(max_tokens):
        token = run_patches(model, run, [patch])["ids"][0]
        if token in stop:
            break
        written.append(token)
        run = add_token(run, token, record=False)
    return run.tokenizer.decode(written)


# ---------------------------------------------------------------------------------------------- other model families
FAMILY_MODELS = {     # the models of Appendix G.2, and the language model each one is built on
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": "Qwen2",
    "OpenGVLab/InternVL3-8B-hf": "Qwen2.5",
    "google/gemma-3-12b-it": "Gemma 3",
    "mistral-community/pixtral-12b": "Mistral",
    "HuggingFaceM4/Idefics3-8B-Llama3": "Llama 3",
}


def _language_model(model):
    """The decoder of a vision-language model: the module with `.layers` and `.norm`."""
    for path in ("model.language_model", "model.text_model", "language_model.model", "language_model"):
        module = model
        for part in path.split("."):
            module = getattr(module, part, None)
        if module is not None and hasattr(module, "layers") and hasattr(module, "norm"):
            return module
    raise RuntimeError("the language model of this model was not found")


def letter_of(answer):
    """The letter an answer token names, in upper case (Idefics3 writes lists in lower case), or None."""
    text = answer.strip()
    return text.upper() if len(text) == 1 and text.isalpha() else None


def name_letter(answer, letters):
    """Which of the named letters an answer token is (case-folded), or "other". `letters` maps a name to a letter."""
    letter = letter_of(answer)
    return next((name for name, named in letters.items() if named == letter), "other")


class VLMPrompt:
    """One tokenized image and question. `positions` names the tokens by their place in the sequence:
        "colon"     the colon token of the prefilled "Answer:"
        "answers"   every written letter;  "last_answer", "last_comma" the last written letter and comma
        "shape"     the shape token of the question (when a shape word is given)
        "image"     every image token
        "last"      the token whose next token is read"""

    def __init__(self, inputs, ids, positions):
        self.inputs, self.ids, self.positions = inputs, ids, positions


class OtherVLM:
    """A vision-language model of another family, in FP32 with eager attention (the block edits the
    attention mask, which only eager attention passes to every layer)."""

    def __init__(self, model_id, device="cuda:0"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from utils.model import REVISIONS
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        revision = REVISIONS.get(model_id)
        self.model_id, self.device = model_id, device
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype=torch.float32, attn_implementation="eager", device_map=device).eval()
        self.model.requires_grad_(False)
        decoder = _language_model(self.model)
        self.layers = list(decoder.layers)
        self.boundaries = self.layers + [decoder.norm]        # residual entering layer d; the last is after the last layer
        config = self.model.config
        self.image_token = getattr(config, "image_token_id", None)
        if self.image_token is None:
            self.image_token = getattr(config, "image_token_index", None)

    @property
    def n_layers(self):
        return len(self.layers)

    def decode(self, token_id):
        return self.tokenizer.decode([token_id])

    def prompt(self, image, question, written=(), shape=None):
        """The model's own chat template with the image first, then the question, then the assistant
        turn prefilled with "Answer:" and the written letters (" X," each)."""
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = chat + (" Answer:" if chat.endswith(":") else "Answer:") + "".join(f" {letter}," for letter in written)
        bos = self.tokenizer.bos_token
        once = not (bos and text.startswith(bos))             # a template that writes the BOS token gets no second one
        inputs = self.processor(text=[text], images=[image.convert("RGB")], return_tensors="pt", add_special_tokens=once)
        inputs = inputs.to(self.device)
        ids = inputs["input_ids"][0].tolist()
        text_of = lambda position: self.decode(ids[position]).strip()
        last = len(ids) - 1
        colon = last - 2 * len(written)
        assert text_of(colon).endswith(":"), f"the colon token was not found: {text_of(colon)!r}"
        answers = [colon + 1 + 2 * i for i in range(len(written))]
        for position, letter in zip(answers, written):
            assert text_of(position) == letter and text_of(position + 1) == ",", "each written letter must be one token"
        positions = {"colon": colon, "answers": answers, "last": last,
                     "image": [i for i, token in enumerate(ids) if token == self.image_token]}
        if written:
            positions.update(last_answer=answers[-1], last_comma=last)
        if shape is not None:
            matches = [i for i in range(colon) if text_of(i).lower() == shape]
            assert matches, f"the shape word {shape!r} is not one token for this model"
            positions["shape"] = matches[-1]
        return VLMPrompt(inputs, ids, positions)

    def run(self, prompt, residual=None, block=(), record=()):
        """One forward pass of the full prompt. Returns (next token as text, recorded residuals).

        residual   (layer, position, vector): the residual entering `layer` at `position` is replaced
                   (layer = number of layers is the input of the final norm)
        block      a list of (tokens, positions): no head of any layer attends from `tokens` to `positions`
        record     positions whose residuals are kept. The second value returned maps each of them
                   to a tensor (layers + 1, hidden size): the residual entering every layer, and
                   the output of the last one."""
        hooks, recorded = [], {position: [None] * len(self.boundaries) for position in record}
        for index, module in enumerate(self.boundaries):
            def before(_module, args, index=index):
                hidden = args[0]
                if residual is not None and residual[0] == index:
                    hidden = hidden.clone()
                    hidden[0, residual[1]] = residual[2].to(hidden.device, hidden.dtype)
                for position in record:
                    recorded[position][index] = hidden[0, position].detach().clone()
                return (hidden,) + tuple(args[1:])
            hooks.append(module.register_forward_pre_hook(before))
        if block:
            def blocked_mask(_module, args, kwargs):
                mask = kwargs.get("attention_mask")
                assert mask is not None and mask.dtype != torch.bool, "a block needs the additive mask of eager attention"
                mask = mask.clone()
                for tokens, positions in block:
                    rows = torch.as_tensor(list(tokens), device=mask.device)
                    columns = torch.as_tensor(list(positions), device=mask.device)
                    mask[:, :, rows[:, None], columns[None, :]] = torch.finfo(mask.dtype).min
                kwargs["attention_mask"] = mask
                return args, kwargs
            hooks += [layer.self_attn.register_forward_pre_hook(blocked_mask, with_kwargs=True) for layer in self.layers]
        try:
            with torch.no_grad():
                logits = self.model(**prompt.inputs, use_cache=False).logits[0, -1]
        finally:
            for hook in hooks:
                hook.remove()
        return self.decode(int(logits.argmax())), {position: torch.stack(states) for position, states in recorded.items()}

    def sweep(self, target, position, vectors):
        """Replace the target's residual at `position` with vectors[layer], one layer at a time, for
        every layer and the output of the last one. Returns the next token of each run."""
        return [self.run(target, residual=(layer, position, vectors[layer]))[0] for layer in range(len(self.boundaries))]

    def generate(self, prompt, max_tokens=16):
        """Greedy writing from the prompt, with no edit. Returns the written text."""
        with torch.no_grad():
            out = self.model.generate(**prompt.inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, len(prompt.ids):], skip_special_tokens=True)

    def image_token_panels(self, image, question, regions):
        """The panel of every image token. Image tokens of other models do not follow one simple grid
        (tiles, a global view, row breaks), so we find the panels by experiment: we paint one panel
        white at a time and give every image token to the panel whose removal changes its input to
        the first layer the most. Returns a list with one panel index per image token."""
        from PIL import ImageDraw
        def first_layer_input(picture):
            prompt = self.prompt(picture, question)
            store = {}
            hook = self.layers[0].register_forward_pre_hook(lambda _m, args: store.update(x=args[0][0, prompt.positions["image"]].float()))
            try:
                with torch.no_grad():
                    self.model(**prompt.inputs, use_cache=False)
            finally:
                hook.remove()
            return store["x"]
        plain = first_layer_input(image)
        changes = []
        for box in regions:
            blank = image.copy()
            ImageDraw.Draw(blank).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill="white")
            changes.append((first_layer_input(blank) - plain).norm(dim=-1))
        return torch.stack(changes, dim=1).argmax(dim=1).tolist()


# ==================================================================================================
# Appendix G: other layouts, prompts, and natural images
# ==================================================================================================
def _finish(run, model, question, prefill, inputs, ids, text, image_positions, record):
    """The part of `Run.__init__` after tokenizing: name the positions, cache the prefix, run once."""
    tokenizer = model.processor.tokenizer
    run.model, run.image, run.question, run.prefill, run.regions = model, None, question, prefill, None
    run.info = {"input_ids": ids, "text": text, "image_positions": image_positions,
                "end_of_turn": tokenizer.convert_tokens_to_ids("<|im_end|>")}
    run.ids, run.tokenizer, run.device = ids, tokenizer, model.device
    run.image_positions = torch.as_tensor(image_positions, device=model.device)
    run.last = len(ids) - 1
    run.names = {"last": run.last}
    run._kv_group = n_heads(model) // model.config.text_config.num_key_value_heads
    run._cache_prefix(inputs)
    run.states = None
    result = _recompute(model, run, [Patch()], record=record)
    run.logits = result["logits"][0]
    run.answer_id = int(run.logits.argmax())
    run.states = result.get("states")


class ImagesRun(Run):
    """A run with several images in one user turn, all before the question. The image tokens of all
    images are `run.image_positions`; the tokens between the images belong to the cached part."""

    def __init__(self, model, images, question, prefill="Answer:", record=True):
        processor = model.processor
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question}]
        text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True) + prefill
        inputs = processor(text=[text], images=[image.convert("RGB") for image in images], return_tensors="pt").to(model.device)
        ids = inputs["input_ids"][0].tolist()
        image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_positions = [i for i, token in enumerate(ids) if token == image_id]
        self.first_text = image_positions[-1] + 1
        _finish(self, model, question, prefill, inputs, ids, text, image_positions, record)


class TextRun(Run):
    """A run with no image. The user turn is the panel descriptions, a space, and the question. The
    cached part ends where the descriptions start, so every token of the user turn can be patched."""

    def __init__(self, model, descriptions, question, prefill="Answer:", record=True):
        processor, tokenizer = model.processor, model.processor.tokenizer
        user = descriptions + " " + question
        messages = [{"role": "user", "content": [{"type": "text", "text": user}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
        encode = lambda part: tokenizer(part, add_special_tokens=False)["input_ids"]
        ids, head = encode(text), text[:text.index(user)]
        start, end = len(encode(head)), len(encode(head + descriptions))
        assert ids[:end] == encode(head + descriptions), "the descriptions must start and end on a token boundary"
        self.first_text = start
        _finish(self, model, question, prefill, {"input_ids": torch.tensor([ids], device=model.device)},
                ids, text, list(range(start, end)), record)

    def _cache_prefix(self, inputs):
        """With no image, the position of a token is its index (the same for the three rotary parts)."""
        model, ids = self.model, inputs["input_ids"]
        positions = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, 1, -1)
        mask = torch.ones_like(ids)
        model.rope_deltas = None
        output = model(input_ids=ids[:, :self.first_text], attention_mask=mask[:, :self.first_text],
                       position_ids=positions[..., :self.first_text], use_cache=True, logits_to_keep=1)
        self._cache = output.past_key_values
        self._text_ids, self._mask, self._positions = ids[:, self.first_text:], mask, positions[..., self.first_text:]
        self._inputs = {"input_ids": ids, "attention_mask": mask, "position_ids": positions}


def nearest_place_positions(run, centers, place):
    """The image tokens of a run that are closer to `centers[place]` than to any other center
    (centers in pixels of the image given to the model). On a canvas this is the widest region
    that can belong to one shape, like a panel with no border."""
    _, grid_h, grid_w = run.info["grid_thw"][0]
    grid_h, grid_w = grid_h // 2, grid_w // 2
    xs = (torch.arange(grid_w, dtype=torch.float32) + 0.5) * run.info["image_width"] / grid_w
    ys = (torch.arange(grid_h, dtype=torch.float32) + 0.5) * run.info["image_height"] / grid_h
    points = torch.stack([xs.repeat(grid_h), ys.repeat_interleave(grid_w)], dim=1)
    nearest = torch.cdist(points, torch.tensor(centers, dtype=torch.float32)).argmin(dim=1)
    return run.image_positions[(nearest == place).to(run.device)]


def run_model_images(model, images, question, prefill="Answer:", record=True):
    """Run the model on several images and one question. See `ImagesRun`."""
    with torch.no_grad():
        return ImagesRun(model, images, question, prefill, record)


def run_model_text(model, descriptions, question, prefill="Answer:", record=True):
    """Run the model on panel descriptions and a question, with no image. See `TextRun`."""
    with torch.no_grad():
        return TextRun(model, descriptions, question, prefill, record)
