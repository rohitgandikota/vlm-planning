"""Load the model and turn an image and a prompt into model inputs.

    from utils.model import load_model, prepare
    model = load_model()                                   # Qwen3-VL-8B-Instruct, FP32
    inputs, info = prepare(model, image, "What is the letter inside the star?", prefill="Answer:")

`info` holds what the experiments need to find their tokens: the token ids, the positions of the
image tokens, and the panel that each image token belongs to.
"""
import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# The exact weights used in the paper. Other model ids load their latest revision.
REVISIONS = {"Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"}


def load_model(model_id=MODEL_ID, device="cuda:0"):
    """Load a Qwen3-VL model the way the paper runs it: FP32 weights, eager attention, TF32 off.

    Eager attention is required: the patches in utils/patching.py edit the attention computation.
    The processor is attached as `model.processor`."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    revision = REVISIONS.get(model_id)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=torch.float32, attn_implementation="eager", device_map=device,
    ).eval()
    model.requires_grad_(False)
    model.processor = processor
    model.model_id = model_id
    return model


def n_layers(model):
    return len(model.model.language_model.layers)


def n_heads(model):
    return model.config.text_config.num_attention_heads


def all_heads(model):
    """Every attention head of the language model as (layer, head)."""
    return [(layer, head) for layer in range(n_layers(model)) for head in range(n_heads(model))]


def prepare(model, image, question, prefill="Answer:"):
    """Tokenize one image and one question. `prefill` is the start of the model's answer.

    The prompt is the chat template with the image first, then the question, then the assistant
    turn, which starts with `prefill`. The model's next token is read after the last prefill token.
    Returns (inputs, info)."""
    processor = model.processor
    tokenizer = processor.tokenizer
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
    inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt").to(model.device)
    ids = inputs["input_ids"][0].tolist()
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_positions = [i for i, token in enumerate(ids) if token == image_id]
    assert image_positions == list(range(image_positions[0], image_positions[-1] + 1)), "image tokens must be one span"
    info = {
        "input_ids": ids,
        "text": text,
        "image_positions": image_positions,
        "grid_thw": inputs["image_grid_thw"].tolist(),
        "image_width": image.size[0],
        "image_height": image.size[1],
        "end_of_turn": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }
    return inputs, info


def letter_token_ids(model):
    """Token id of each capital letter written after a space (" A", " B", ...), as the model answers."""
    tokenizer = model.processor.tokenizer
    ids = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        tokens = tokenizer.encode(" " + letter, add_special_tokens=False)
        assert len(tokens) == 1, letter
        ids[letter] = tokens[0]
    return ids


def panel_of_image_tokens(info, regions):
    """For every image token, the index of the region (panel) it lies in, or -1 for none.

    `regions` is a list of [x0, y0, x1, y1] boxes in pixels of the image given to the model.
    The model merges 2x2 patches, so the token grid is half the patch grid on each side."""
    _, grid_h, grid_w = info["grid_thw"][0]
    grid_h, grid_w = grid_h // 2, grid_w // 2
    assert grid_h * grid_w == len(info["image_positions"]), "image grid does not match the image tokens"
    width, height = info["image_width"], info["image_height"]
    xs = np.tile((np.arange(grid_w) + 0.5) * width / grid_w, grid_h)
    ys = np.repeat((np.arange(grid_h) + 0.5) * height / grid_h, grid_w)
    mapping = np.full(len(xs), -1, dtype=np.int64)
    for index, (x0, y0, x1, y1) in enumerate(regions):
        mapping[(xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)] = index
    return mapping
