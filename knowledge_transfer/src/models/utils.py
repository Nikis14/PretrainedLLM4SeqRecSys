import torch
from torch import nn
import torch.nn.functional as F


def mean_weightening(hidden_states):
    """hidden_states: [batch_size, seq_len, hidden_size]"""
    return hidden_states.mean(dim=1)


def exponential_weightening(hidden_states, weight_scale):
    """hidden_states: [batch_size, seq_len, hidden_size]"""
    device = hidden_states.device

    indices = torch.arange(hidden_states.shape[1]).float().to(device)  # [0, 1, 2, ..., seq_len-1]
    weights = torch.exp(weight_scale * indices)  # Shape: [seq_len]

    # Normalize weights (optional, for scale invariance)
    weights = weights / weights.sum()

    # Reshape weights to [1, n_items, 1] for broadcasting
    weights = weights.view(1, hidden_states.shape[1], 1)

    # Apply weights and aggregate
    weighted_tensor = hidden_states * weights
    result = weighted_tensor.sum(dim=1)  # Aggregated tensor, shape: [batch_size, hidden_units]
    return result


class SimpleAttentionAggregator(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleAttentionAggregator, self).__init__()
        self.attention = nn.Linear(hidden_units, 1)  # Learnable attention weights

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, n_items, hidden_units]
        Returns:
        Aggregated tensor of shape [batch_size, hidden_units]
        """
        # Compute attention scores (shape: [batch_size, n_items, 1])
        scores = self.attention(x)

        # Normalize scores with softmax over the 2nd dimension
        weights = F.softmax(scores, dim=1)  # Shape: [batch_size, n_items, 1]

        # Weighted sum of the input tensor
        weighted_sum = (x * weights).sum(dim=1)  # Shape: [batch_size, hidden_units]
        return weighted_sum


def last_item_weightening(hidden_states):
    """hidden_states: [batch_size, seq_len, hidden_size]"""
    return hidden_states[:, -1, :]
