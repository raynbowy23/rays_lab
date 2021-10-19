import types
from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from icecream import ic
ic.configureOutput(includeContext=True)


class MultiHeadAttention(hk.Module):
    """
    Split several small heads and use attention has prior efficiency to single attention.
    First, split query, key, and value to head_num and calculate attention,
    then concatenate lastly.

    Attributes:
        depth: the dimension of hidden layer and output
    """ 
    def __init__(self, num_heads: int, key_size: int, w_init_scale: float, value_size: Optional[int] = None, model_size: Optional[int] = None, dropout_rate: float = 0.0, name: Optional[str] = None):
        """Inits scaled_attention"""
        super().__init__(name=name)
        self.head_num = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * self.head_num
        self.w_init = hk.initializers.VarianceScaling(w_init_scale)
        self.scale_factor = np.sqrt(key_size) #.astype(key_size.dtype)
        self.dropout_rate = dropout_rate
        

    def __call__(self, query: jnp.ndarray, key: Optional[jnp.ndarray] = None, value: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None, name: Optional[str] = None) -> jnp.ndarray:

        query_heads = self._linear_projection(query, self.key_size, "query") # head_size -> key_size # WQ
        key_heads = self._linear_projection(key, self.key_size, "key") # WK
        value_heads = self._linear_projection(value, self.value_size, "value") # WV

        # scale dot product
        # [bath, heads, tokens, tokens]
        attn_logits = jnp.einsum('...thd, ...Thd -> ...htT', query_heads, key_heads) # [batch_size, head_num, q_length, hidden_dim/head_num]

        # mask
        sqrt_key_size = np.sqrt(self.key_size).astype(key.dtype)
        # attn_logits = attn_logits.masked_fill(attention_mask==0, -1e30)
        attn_logits = attn_logits / sqrt_key_size  
        if mask is not None:
            assert mask.shape == attn_logits.shape[-2:]
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        attention_weights = jax.nn.softmax(attn_logits)
        # TODO: attention_weight = dropout_layer
        # TODO: add Dropout
        # [batch_size, head_num, q_length, hidden_dim/head_num]
        attn = jnp.einsum("...htT, ...Thd -> ...thd", attention_weights, value_heads) # [batch_size, q_length, hidden_dim]
        # Concatenate attention matrix of all heads into a single vector
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))

        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec) # WO

    @hk.transparent
    def _linear_projection(
        self,
        x: jnp.ndarray,
        head_size: int,
        name: Optional[str] = None
    ) -> jnp.ndarray:
        y = hk.Linear(self.head_num * head_size, w_init=self.w_init, name=name)(x)
        return y.reshape((*x.shape[:-1], self.head_num, head_size))


class SelfAttention(MultiHeadAttention):
    """
    Self attention with a causal mask applied

    Attributes:
        query: input
        key: <- memory
        value: <- memory
        mask: attention mask

    Return:
        input -> memory
    """
    def __call__(self, query: jnp.ndarray, key: Optional[jnp.ndarray] = None, value: Optional[jnp.ndarray] = None, attention_mask: Optional[jnp.ndarray] = None, name: Optional[str] = None) -> jnp.ndarray:
        key = key if key is not None else query
        value = value if value is not None else query
        # memory = memory if memory is not None else query

        seq_len = query.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        attention_mask = attention_mask * causal_mask if attention_mask is not None else causal_mask

        return super().__call__(query=query, key=key, value=value, mask=attention_mask, name=name)