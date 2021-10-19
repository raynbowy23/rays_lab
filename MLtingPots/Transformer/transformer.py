import os
import torch
import torch.nn as nn

import types
from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from icecream import ic
ic.configureOutput(includeContext=True)

from scale_attention import ScaledAttention
from multihead_attention import SelfAttention, MultiHeadAttention # SelfAttention
from embedding import AddPositionalEncoding, TokenEmbedding
from ffn import FeedForwardNetwork, ResidualNormalizationWrapper
from metrics import padded_cross_entropy_loss, padded_accuracy

# def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
#     """Apply a unique LayerNorm to x with default settings"""
#     return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x) # TODO: implementat later

class Encoder(hk.Module):
    """
    Encoder which encodes token sequences to vector sequences
    """
    def __init__(self, vocab_size: int, hopping_num: int, head_num: int, input_in_feature: int, memory_in_feature, hidden_dim: int, dropout_rate: float, max_length: int) -> None:
        super().__init__()
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.input_in_feature = input_in_feature
        self.memory_in_feature = memory_in_feature
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.init_scale = 2. / hopping_num

        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        # self.input_dropout_layer = torch.nn.Dropout(dropout_rate)
        # self.input_dropout_layer = hk.dropout(rng, dropout_rate)

        self.attention_block_list: List[List[nn.Module]] = []
        # self.h_norm = layer_norm(self.token_embedding, name=f'h_layer_norm')

        for i in range(hopping_num):
            attention_layer = SelfAttention(num_heads=self.head_num, key_size=input_in_feature, w_init_scale=self.init_scale, name=f'h{i}_attn') # .cuda() # (h_norm)
            ffn_layer = FeedForwardNetwork(hidden_dim=hidden_dim, init_scale=self.init_scale, name=f'h{i}_ffn') # .cuda()
            self.attention_block_list.append([
                ResidualNormalizationWrapper(attention_layer, dropout_rate),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate),
            ])

    def __call__(self, input: jnp.ndarray, self_attention_mask: Optional[jnp.ndarray], training: bool) -> jnp.ndarray:
        """
        Execute model.

        Args:
            input: shape = [batch_size, length]
            training: True if train
            shape: [batch_size, length, hidden_dim]
        """
        rng = jax.random.PRNGKey(42)
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = hk.dropout(rng, rate=self.dropout_rate, x=embedded_input)

        for i, model in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = tuple(model)
            query = attention_layer(query, memory=None, attention_mask=self_attention_mask, training=training)
            # query = attention_layer(query, memory=query, attention_mask=None, training=training)
            query = ffn_layer(query, memory=None, attention_mask=None, training=training)
            # layer_norm = torch.nn.LayerNorm(query.size()[1:]).cuda()
            layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(query)
            # [batch_size, length, hidden_dim]
            return layer_norm


class Decoder(hk.Module):
    """
    Decoder which generates token sequences from encoded vector sequences
    """
    def __init__(self, vocab_size: int, hopping_num: int, head_num: int, input_in_feature: int, memory_in_feature: int, hidden_dim: int, dropout_rate: float, max_length: int) -> None:
        super().__init__()
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.input_in_feature = input_in_feature
        self.memory_in_feature = memory_in_feature
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.init_scale = 2. / hopping_num

        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        # self.input_dropout_layer = torch.nn.Dropout(dropout_rate)
        # self.input_dropout_layer = hk.dropout(dropout_rate)
        # if mask is not None:
        #     mask = mask[:, None, None, :]
        # self.h_norm = layer_norm(self.add_position_embedding, name=f'h_layer_norm')

        self.attention_block_list: List[List[nn.Module]] = []
        for i in range(hopping_num):
            # self_attention_layer = SelfAttention(input_in_feature, memory_in_feature, hidden_dim, head_num, dropout_rate).cuda()
            self_attention_layer = SelfAttention(num_heads=self.head_num, key_size=input_in_feature, w_init_scale=self.init_scale, name=f'h{i}_attn_dec') # .cuda() #(self.h_norm)
            enc_dec_attention_layer = MultiHeadAttention(head_num, key_size=input_in_feature, w_init_scale=self.init_scale, value_size=memory_in_feature, model_size=hidden_dim, name=f'h{i}_mattn') # .cuda() #dropout_rate=dropout_rate, name=f'h{i}_mattn') # .cuda()
            ffn_layer = FeedForwardNetwork(hidden_dim=hidden_dim, init_scale=self.init_scale, name=f'h{i}_ffn_dec') # .cuda()
            self.attention_block_list.append([
                ResidualNormalizationWrapper(self_attention_layer, dropout_rate),
                ResidualNormalizationWrapper(enc_dec_attention_layer, dropout_rate),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate)
            ])
        # self.output_dense_layer = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        self.w_init = hk.initializers.VarianceScaling(self.init_scale)
        self.output_dense_layer = hk.Linear(head_num, vocab_size, self.w_init)

    def __call__(self, input: jnp.ndarray, encoder_output: jnp.ndarray, self_attention_mask: jnp.ndarray, enc_dec_attention_mask: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Execute model.

        Args:
            input: [batch_size, length]
            training: True when trains
        
        Returns:
            decoded query [batch_size, length, vocab_size]
        """
        rng = jax.random.PRNGKey(42)
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = hk.dropout(rng, rate=self.dropout_rate, x=embedded_input)

        for i, model in enumerate(self.attention_block_list):
            self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(model)

            # query = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(query)
            query = self_attention_layer(query, memory=None, attention_mask=self_attention_mask, training=training)
            # query = attention_layer(query, memory=None, attention_mask=self_attention_mask, training=training) #, mask=self_attention_mask)
            # h = h + h_attn

            query = enc_dec_attention_layer(query, memory=encoder_output, attention_mask=enc_dec_attention_mask, training=training)
            # h = h + h_enc_dec

            query = ffn_layer(query, memory=None, attention_mask=None, training=training)
            # query = self.dropout_layer(query) # TODO: implement dropout
            # h = h + h_ffn

        # layer_norm = torch.nn.LayerNorm(query.size()[1:]).cuda()
        layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(query)
        # query = layer_norm(query) # [batch_size, length, hidden_dim]
        return self.output_dense_layer(layer_norm) # [batch_size, length, vocab_size]

# PAD_ID = torch.tensor(0)
PAD_ID = jnp.array(0)

class Transformer(hk.Module):
    """
    A Transformer stack
    """
    def __init__(self, vocab_size: int, hopping_num: int=4, head_num: int=8, input_in_feature: int=64, memory_in_feature: int=64, hidden_dim: int=512, dropout_rate: float=0.1, max_length: int=512) -> None:
        # super(Transformer, self).__init__()
        super().__init__(name="Transformer")
        self.vocab_size = vocab_size
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.memory_in_feature = memory_in_feature
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.input_in_feature = input_in_feature

        self.encoder = Encoder(vocab_size, hopping_num, head_num, self.input_in_feature, self.memory_in_feature, hidden_dim, dropout_rate, max_length)
        self.decoder = Decoder(vocab_size, hopping_num, head_num, self.input_in_feature, self.memory_in_feature, hidden_dim, dropout_rate, max_length)
        # self.loss_identity = torch.nn.Identity() # a placeholder identity operator that is argument-insensitive
        # self.acc_identity = torch.nn.Identity()
        # self.loss_identity = jnp.identity # a placeholder identity operator that is argument-insensitive
        # self.acc_identity = jnp.identity

    def __call__(self, encoder_input: jnp.ndarray, decoder_input: jnp.ndarray, training: bool) -> jnp.ndarray:
        enc_attention_mask = self._create_enc_attention_mask(encoder_input)
        dec_self_attention_mask = self._create_dec_self_attention_mask(decoder_input)

        encoder_output = self.encoder(input=encoder_input, self_attention_mask=enc_attention_mask, training=training)

        # decoder_output = self.decoder(input=decoder_input, encoder_output=encoder_output,
        #     self_attention_mask=dec_self_attention_mask, enc_dec_attention_mask=enc_attention_mask, training=training)
        output_embeddings = self.decoder(input=decoder_input, encoder_output=encoder_output,
            self_attention_mask=dec_self_attention_mask, enc_dec_attention_mask=enc_attention_mask, training=training)

        # decoder_target = decoder_input[:, 1:] # without BOS

        # xentropy, weights = padded_cross_entropy_loss(
        #     decoder_output, decoder_target, smoothing=0.05, vocab_size=self.vocab_size
        # )
        # loss = self.loss_identity(jnp.sum(xentropy) / jnp.sum(weights))

        # accuracies, weights = padded_accuracy(decoder_output, decoder_target)
        # acc = self.acc_identity(jnp.sum(accuracies) / jnp.sum(weights))

        # return decoder_output
        return output_embeddings

    def _create_enc_attention_mask(self, encoder_input: jnp.ndarray):
        '''Ignores <PAD>
        '''
        batch_size, length = encoder_input.shape
        # pad_array = torch.eq(torch.from_numpy(encoder_input.clone().detach()).requires_grad_(True), PAD_ID) # [batch_size, m_length]
        # pad_array = jnp.equal(encoder_input.clone().detach(), PAD_ID) # [batch_size, m_length]
        pad_array = jnp.equal(encoder_input, PAD_ID) # [batch_size, m_length]
        # shape broadcasting
        return jnp.reshape(pad_array, [batch_size, 1, 1, length]) # .repeat(self.head_num, 1)

    def _create_dec_self_attention_mask(self, decoder_input: jnp.ndarray):
        '''Ignores the future info
        '''
        batch_size, length = decoder_input.shape
        # pad_array = jnp.equal(decoder_input.clone().detach(), PAD_ID) # [batch_size, m_length]
        pad_array = jnp.equal(decoder_input, PAD_ID) # [batch_size, m_length]
        # pad_array = torch.eq(torch.from_numpy(decoder_input.clone().detach()).requires_grad_(True), PAD_ID) # [batch_size, m_length]
        pad_array = jnp.reshape(pad_array, [batch_size, 1, 1, length])
        # pad_array = jnp.einsum('bl -> b..l', pad_array) # pad_array, [batch_size, 1, 1, length])

        # if args.cuda:
        autoregression_array = jnp.logical_not(jnp.triu(jnp.ones([length, length], dtype=jnp.bool_), k=-1)) # .cuda()
        # else:
        # autoregression_array = torch.logical_not(torch.triu(torch.ones([length, length], dtype=torch.bool), diagonal=-1))
        autoregression_array = jnp.reshape(autoregression_array, [1, 1, length, length])
        # autoregression_array = jnp.einsum('ll -> ...ll', autoregression_array) # .reshape(autoregression_array, [1, 1, length, length])

        return jnp.logical_or(pad_array, autoregression_array)
