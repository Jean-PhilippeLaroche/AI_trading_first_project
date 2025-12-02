import matplotlib.pyplot as plt
import numpy as np

# Global figure (persists across epochs)
_fig = None
_axes = None
_initialized = False


def init_attention_window(num_layers, n_heads, seq_len):
    """
    Creates a persistent window showing attention heatmaps.
    One subplot per head per layer.
    """

    global _fig, _axes, _initialized

    if _initialized:
        return

    total_plots = num_layers * n_heads

    cols = n_heads
    rows = num_layers

    _fig, _axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1 and cols == 1:
        _axes = np.array([[ _axes ]])
    elif rows == 1:
        _axes = np.array([_axes])
    elif cols == 1:
        _axes = np.array([[_axes[i]] for i in range(rows)])

    for layer in range(num_layers):
        for head in range(n_heads):
            ax = _axes[layer][head]
            ax.set_title(f"Layer {layer+1}, Head {head+1}")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.ion()
    plt.show()

    _initialized = True



def update_attention_window(mean_attn, epoch):
    """
    mean_attn: list of [heads, seq, seq] arrays (one per layer)
    updates the live attention heatmap window in real time
    """

    global _fig, _axes
    if _fig is None:
        return  # window not created yet

    num_layers = len(mean_attn)
    num_heads = mean_attn[0].shape[0]

    for layer in range(num_layers):
        for head in range(num_heads):
            ax = _axes[layer][head]
            ax.clear()
            ax.set_title(f"Layer {layer+1}, Head {head+1} (Epoch {epoch})")
            ax.imshow(mean_attn[layer][head], cmap='viridis', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

    _fig.canvas.draw()
    _fig.canvas.flush_events()


if __name__ == "__main__":
    pass