import unittest
import jax
import jax.numpy as jnp
import numpy as np
from Transformer.multihead_attention import MultiheadAttention, SelfAttention

class TestMultiheadAttention(unittest.TestCase):
    def test_call(self):
        batch_size = 3
        q_length = 5
        m_length = 7
        hidden_dim = 32
        num_heasd = 4
        input_in_feature=64
        memory_in_feature=64
        hopping_num = 4
        init_scale = 2. / hopping_num

        q = jnp.ndarray([None, None, hidden_dim])
        k = jnp.ndarray([None, None, hidden_dim])

        mask_numpy = np.zeros(shape=[batch_size, 1, q_legth, m_length])
        mask_numpy[0, 0, :, -1] = 1

        model = MultiheadAttention(num_heads=num_heads, key_size=input_in_feature, w_init_scale=init_scale, value_size=memory_in_feature, model_size=hidden_dim, dropout_rate=0.1)
        result_op = model(q, k, mask, training=True)
        # RUN
        # result, attention_weight = RUN
        self.assertEqual(result.shape, (batch_size, q_length, hidden_dim))
        self.assertEqual(attention_weight[0, 0, :, -1].tolist(), [0.0] * q_length)

