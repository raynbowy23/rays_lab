import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import math

class AddPositionalEncoding(hk.Module):
    """
    The layer that returns position information corresponding to input tensor.
    see: https://arxiv.org/pdf/1706.03762.pdf

    PE_{pos, 2i} = sin(pos / 10000^{2i / d_model})
    PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
    where d_model is the embeddings dimension.
    """
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        fl_type = inputs.dtype
        batch_size, max_length, depth = inputs.shape
        depth_counter = jnp.arange(depth) // 2 * 2
        # depth_matrix = torch.unsqueeze(depth_counter, 0).repeat([max_length, 1]) # [max_length, depth]
        depth_matrix = jnp.expand_dims(depth_counter, 0).repeat(max_length, 0)
        depth_matrix = jax.lax.pow(10000.0, (depth_matrix / depth)) #.promote_types(fl_type)) # [max_length, depth]

        # cos(x) == sin(x + pi/2)
        phase = (jnp.arange(depth) % 2) * math.pi / 2 # 0, pi/2, 0, pi/2, ...
        phase_matrix = jnp.expand_dims(phase, 0).repeat(max_length, 0)

        pos_counter = jnp.arange(max_length)
        pos_matrix = jnp.expand_dims(pos_counter, 1).repeat(depth, 1)

        positional_encoding = jnp.sin(pos_matrix / depth_matrix + phase_matrix)
        positional_encoding = jnp.expand_dims(positional_encoding, 0).repeat(batch_size, 0)

        return inputs + positional_encoding


PAD_ID = jnp.array(0)

class TokenEmbedding(hk.Module):
    """
    Transforms each token (int) to embedded vector

    Attributes:
        vocab_size: vocabulary size
        embedding_dim: embedding dimension
        dtype: vector type

    Returns:
        embedding vector
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        self.embedding = hk.Embed(vocab_size=self.vocab_size, embed_dim=self.embedding_dim, w_init=self.embed_init) # lookup table


    def __call__(self, input:jnp.ndarray) -> jnp.ndarray:
        sk = jnp.greater(input, PAD_ID)
        embedding = self.embedding(input)
        return embedding * self.embedding_dim ** 0.5