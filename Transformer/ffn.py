import types
from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from icecream import ic
ic.configureOutput(includeContext=True)

class FeedForwardNetwork(hk.Module):
    """
    Position-wise Feedforward Neural Network for Transformer
    """
    def __init__(self, hidden_dim: int, init_scale: float, widening_factor: int = 4, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor
        self.hidden_dim = hidden_dim

    def __call__(self, x: jnp.ndarray, key=None, value=None) -> jnp.ndarray:
        """Apply FeedForwardNetwork
        Args:
            input: shape = [batch_size, length, input_dim]
        Returns:
            tensor: shape = [batch_size, length, hidden_dim]
        """
        initializer = hk.initializers.VarianceScaling(self._init_scale) # TODO: Understand
        x = hk.Linear(self._widening_factor * self.hidden_dim, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        # TODO: Dropout
        x = hk.Linear(self.hidden_dim, w_init=initializer)(x)
        return x


class ResidualNormalizationWrapper(hk.Module):
    """
    Inputs Residual Connection.
    It consists of Layer Normalization, any layer info, and Dropout.

    Attrinbutes:
        layer: any layer info
        dropout_rate: dropout rate (probability p). It randomly zeroes some of the elements of the input tensor

    Returns:
        Sum of input and calculated tensor
    """

    def __init__(self, layer, dropout_rate: float) -> None:
        super().__init__()
        self.layer = layer
        self.dropout_rate = dropout_rate

    def __call__(self, input: jnp.ndarray, memory: jnp.ndarray, attention_mask: jnp.ndarray, training: bool) -> jnp.ndarray:
        # layer_norm = torch.nn.LayerNorm(input.size()[1:]).cuda() # input_in_feature, hidden_size
        tensor = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(input) # https://arxiv.org/abs/1607.06450
        tensor = self.layer(tensor, key=memory, value=memory)
        # tensor = self.dropout_layer(tensor) # TODO: implement dropout
        tensor = hk.dropout(hk.next_rng_key(), rate=self.dropout_rate, x=tensor)
        return input + tensor