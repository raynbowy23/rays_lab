import numpy as np
import jax
import jax.numpy as jnp

from batch_generator import BatchGenerator
from transformer import Transformer
from metrics import _pad_tensors_to_same_length

from icecream import ic
ic.configureOutput(includeContext=True)

def make_lr_schedule(warmup_percentage, total_steps):
    def lr_schedule(step):
        percent_complete = step / total_steps
        before_peak = jax.lax.convert_element_type(
            (percent_complete <= warmup_percentage),
            np.float32
        )
        scale = (
            (
                before_peak * (percent_complete / warmup_percentage) + (1 - before_peak)
            ) * (1 - percent_complete)
        )
        return scale

    return lr_schedule

# @functools.partial(jax.jit, static_argnums=0)
@jax.jit
def accuracy(predictions, params, rng, data):
    batch_labels = jnp.array(data['transformer/decoder_input:0']) # target
    predictions, batch_labels = _pad_tensors_to_same_length(predictions, batch_labels)

    pre = jnp.argmax(predictions, axis=-1)
    res = jnp.mean(jnp.equal(pre, batch_labels))

    return res

def measure_current_performance(logits, dataset, params, rng, n_examples=None, splits=('train', 'test')):
    # Load our training evaluation and test evaluation splits

    if 'train' in splits:
        # Compute mean train accuracy
        acc = []
        for i, train_eval_batch in enumerate(dataset):
            acc.append(accuracy(
                logits,
                params,
                rng,
                train_eval_batch
            ))
            if i == n_examples:
                break

        train_accuracy = np.mean(acc)
        # train_accuracy = jnp.mean([
        #     accuracy(
        #         logits,
        #         params,
        #         rng,
        #         train_eval_batch
        #     )
        #     for train_eval_batch in dataset
        # ])
        print(f"\t Train validation acc: {train_accuracy:.3f}")

    if 'test' in splits:
        # Compute mean test accuracy
        test_accuracy = jnp.mean([
            accuracy(
                logits,
                params,
                rng,
                test_eval_batch
            )
            for test_eval_batch in dataset
        ])
        print(f"\t Test validation accuracy: {test_accuracy:.3f}")