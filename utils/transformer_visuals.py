import matplotlib.pyplot as plt
import numpy as np
import torch

# Global figure (persists across epochs)
_fig = None
_axes = None
_initialized = False

# Feature importance tracking
_feature_importance_history = []
_feature_names = None


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

    _fig, _axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    if rows == 1 and cols == 1:
        _axes = np.array([[_axes]])
    elif rows == 1:
        _axes = np.array([_axes])
    elif cols == 1:
        _axes = np.array([[_axes[i]] for i in range(rows)])

    for layer in range(num_layers):
        for head in range(n_heads):
            ax = _axes[layer][head]
            ax.set_title(f"Layer {layer + 1}, Head {head + 1}")
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
            ax.set_title(f"Layer {layer + 1}, Head {head + 1} (Epoch {epoch})")
            ax.imshow(mean_attn[layer][head], cmap='viridis', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

    _fig.canvas.draw()
    _fig.canvas.flush_events()


def set_feature_names(feature_names):
    """
    Set the names of features for importance tracking.
    Should be called once before training.

    Args:
        feature_names: List of feature names (e.g., ['RSI', 'MACD', 'close', 'SMA'])
    """
    global _feature_names, _feature_importance_history
    _feature_names = feature_names
    _feature_importance_history = []


def compute_feature_importance(batch_X, attn_weights):
    """
    Compute per-feature attention importance from attention weights.

    This works by:
    1. Taking attention weights across all heads and layers
    2. Aggregating attention scores for each timestep
    3. Using the input embeddings to attribute attention back to original features

    Args:
        batch_X: Input batch tensor of shape (batch, seq_len, num_features)
        attn_weights: List of attention tensors from each layer
                     Each tensor has shape (batch, heads, seq_len, seq_len)

    Returns:
        feature_importance: numpy array of shape (num_features,) with importance scores
    """
    if len(attn_weights) == 0:
        return None

    batch_size, seq_len, num_features = batch_X.shape

    # Stack all attention weights: (num_layers, batch, heads, seq_len, seq_len)
    stacked_attn = torch.stack(attn_weights, dim=0)

    # Average across layers and heads: (batch, seq_len, seq_len)
    avg_attn = stacked_attn.mean(dim=(0, 2))

    # For each position, sum the attention it receives from all other positions
    # This gives us attention importance per timestep: (batch, seq_len)
    timestep_importance = avg_attn.sum(dim=1)

    # Average across batch
    timestep_importance = timestep_importance.mean(dim=0)  # (seq_len,)

    # Now attribute timestep importance to features
    # Use absolute values of input features as a proxy for their contribution
    # Weight each feature by the attention at its timestep

    # Get absolute feature values averaged across batch: (seq_len, num_features)
    feature_magnitudes = batch_X.abs().mean(dim=0)

    # Weight each feature by timestep importance: (seq_len, num_features)
    weighted_features = feature_magnitudes * timestep_importance.unsqueeze(-1)

    # Sum across timesteps to get per-feature importance: (num_features,)
    feature_importance = weighted_features.sum(dim=0)

    # Normalize to sum to 1
    feature_importance = feature_importance / (feature_importance.sum() + 1e-8)

    return feature_importance.cpu().numpy()


def log_feature_importance_to_tensorboard(writer, batch_X, attn_weights, epoch):
    """
    Compute feature importance and log to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        batch_X: Input batch tensor of shape (batch, seq_len, num_features)
        attn_weights: List of attention tensors from each layer
        epoch: Current epoch number
    """
    global _feature_names, _feature_importance_history

    if writer is None or _feature_names is None:
        return

    # Compute feature importance
    feature_importance = compute_feature_importance(batch_X, attn_weights)

    if feature_importance is None:
        return

    # Store history
    _feature_importance_history.append(feature_importance)

    # Log individual feature importances
    for i, feature_name in enumerate(_feature_names):
        writer.add_scalar(f'FeatureImportance/{feature_name}',
                          feature_importance[i],
                          epoch)

    # Log all features as a bar chart (using scalars with common prefix)
    # This will be grouped nicely in TensorBoard
    for i, feature_name in enumerate(_feature_names):
        writer.add_scalar(f'FeatureImportance_Summary/{feature_name}',
                          feature_importance[i],
                          epoch)


def get_feature_importance_history():
    """
    Get the history of feature importances across epochs.

    Returns:
        numpy array of shape (num_epochs, num_features)
    """
    global _feature_importance_history
    if len(_feature_importance_history) == 0:
        return None
    return np.array(_feature_importance_history)


def reset_feature_tracking():
    """
    Reset feature importance tracking.
    Call this at the start of a new training run.
    """
    global _feature_importance_history
    _feature_importance_history = []


if __name__ == "__main__":
    pass