"""Small plots for the notebook: show an image, and show where attention heads look.

    from utils.plotting import show, show_attention, show_text_attention
"""
import matplotlib.pyplot as plt
import numpy as np


def show(*images, titles=None, height=2.2):
    """Show one or more strips under each other."""
    fig, axes = plt.subplots(len(images), 1, figsize=(height * images[0].width / images[0].height, height * len(images)))
    for i, (ax, image) in enumerate(zip(np.atleast_1d(axes), images)):
        ax.imshow(image)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i], fontsize=10, loc="left")
    plt.tight_layout()
    plt.show()


def show_attention(run, heads, token, title=None, height=2.2):
    """Where a set of heads look in the image at one token: their summed attention over the image
    tokens, drawn over the image. Also prints the share of that attention on each panel."""
    _, grid_h, grid_w = run.info["grid_thw"][0]
    grid_h, grid_w = grid_h // 2, grid_w // 2
    total = sum(run.attention(layer, head, token)[run.image_positions] for layer, head in heads)
    heat = total.reshape(grid_h, grid_w).float().cpu().numpy()
    image = run.image
    fig, ax = plt.subplots(figsize=(height * image.width / image.height, height))
    ax.imshow(image)
    ax.imshow(heat, cmap="Blues", alpha=0.75, extent=(0, image.width, image.height, 0), interpolation="nearest")
    ax.axis("off")
    ax.set_title(title or f"attention at the {token!r} token", fontsize=10, loc="left")
    plt.tight_layout()
    plt.show()
    shares = run.panel_attention(heads, token)
    print("share of image attention per panel:", "  ".join(f"{100 * s:.0f}%" for s in shares))


def show_text_attention(run, heads, token, title=None):
    """Which text tokens a set of heads read at one token: their mean attention to each earlier
    text token after the image, as a bar chart."""
    position = run.token(token)
    weights = sum(run.attention(layer, head, position) for layer, head in heads) / len(heads)
    tokens = [(p, t) for p, t in run.text_tokens() if p <= position]
    values = [float(weights[p]) for p, _ in tokens]
    fig, ax = plt.subplots(figsize=(max(6, 0.42 * len(tokens)), 2.4))
    ax.bar(range(len(tokens)), values, color="#0072B2")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([repr(t)[1:-1] for _, t in tokens], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("attention")
    ax.set_title(title or f"text tokens read at the {token!r} token", fontsize=10, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()
