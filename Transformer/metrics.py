import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from icecream import ic
ic.configureOutput(includeContext=True)

# copy and edit to be worked with pytorh from https://github.com/tensorflow/models/blob/master/official/transformer/utils/metrics.py

def padded_cross_entropy_loss(logits, labels, smoothing: float, vocab_size):
    """
    Calculates cross entropy loss while ignoring padding.

    Args:
        logits: tensor of logits [batch_size, length_logits, vocab_size]
        labels: tensor of labels [batch_size, length_labels]
        smoothing: label smoothing constant, used to determine the switching values
        vocab_size: int size of the vocabulary

    Returns:
        Returns the cross entropy loss and weight tensors: float32 tensors with
        shape [batch_size, max(length_logits, length_labels)]
    """
    logits, labels = _pad_tensors_to_same_length(logits, labels)

    # Calculate smoothing cross entropy
    confidence = 1.0 - smoothing
    low_confidence = (1.0 - confidence) / float(vocab_size - 1)
    soft_targets = jnp.clip(hk.one_hot(labels, labels.shape[1]), a_min=low_confidence, a_max=confidence) # vocab_size = config['n_classes']
    loss = cross_entropy(logits, soft_targets)

    # Calculate the best (lowest) possible value of cross entropy, and 
    # subtract from the cross entropy loss
    normalizing_constant = -(
        confidence * jnp.log(confidence) + float(vocab_size - 1) *
        low_confidence * jnp.log(low_confidence + 1e-20)
    )
    loss -= normalizing_constant

    weights = jnp.not_equal(labels, 0) # .clone().detach().float()
    return loss * weights, weights

def cross_entropy(logprobs, targets):
    target_class = np.argmax(targets, axis=1)
    ic(logprobs.shape)
    ic(targets.shape)
    ic(target_class.shape)
    nll = -target_class * np.log(logprobs) - (1 - target_class) * np.log(1 - logprobs)
    loss = jnp.mean(nll)
    return loss
    # nll = jnp.take_along_axis(logprobs, target_class, axis=-1)
    # ce = -jnp.mean(nll)
    # return ce

def to_one_hot_vector(label, num_class):
    b = jnp.zeros(label.shape[0], 8000).to(jnp.long)
    label = label.to(jnp.long).permute(1,0)
    b[range(b.shape[0]), label] = 1

    return b

def padded_accuracy(logits, labels):
    """Percentage of times that predictions matches labels on non-0s."""
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = jnp.not_equal(labels, 0).to(jnp.float32)
    outputs = jnp.argmax(logits, axis=-1).to(jnp.int32)
    padded_labels = labels.to(jnp.int32)
    return jnp.ndarray(jnp.equal(outputs, padded_labels), dtype=jnp.float32), weights

def _pad_tensors_to_same_length(x, y):
    """Pads x and y so that the results have the same length (second dimension)"""
    x_length = x.shape[1]
    y_length = y.shape[1]

    max_length = max(x_length, y_length)

    x = jnp.pad(x, pad_width=([0, 0], [0, max_length - x_length], [0, 0])) # , [0, 0]]) # padding length must be divided by 2
    y = jnp.pad(y, pad_width=([0, 0], [0, max_length - y_length]))
    return x, y