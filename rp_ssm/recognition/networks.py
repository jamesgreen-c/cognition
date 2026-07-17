import flax.linen as nn

from jax import Array
from typing import Sequence


class MLP(nn.Module):
    """Simple MLP with no final Dense layer."""
    features : Sequence[int]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        return x


class CNN(nn.Module):
    """Simple CNN with no final Dense layer."""
    cnn_features : Sequence[dict]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.cnn_features:
            x = nn.Conv(**feat)(x)
            x = nn.relu(x)
        return x.flatten()

