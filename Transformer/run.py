import functools
import optax
from typing import Mapping, Any

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from icecream import ic

from transformer import Transformer
from utils import make_lr_schedule, measure_current_performance
from batch_generator import encode_batch, BatchGenerator, load_dataset
from metrics import padded_cross_entropy_loss
from config import config

# setup logger
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


def transformer_fn(encoder_input, decoder_input, is_training: bool = True):
    def forward_fn(data: Mapping[str, jnp.ndarray] = None) -> jnp.ndarray:
        # TODO: Replace with config
        output_embeddings = Transformer(vocab_size=8000, hopping_num=4, head_num=8, input_in_feature=64, memory_in_feature=64, hidden_dim=512, dropout_rate=0.1, max_length=200)(encoder_input, decoder_input, is_training)
        return hk.Linear(output_size=8000)(output_embeddings) # WO

    return forward_fn

def lm_loss_fn(transformer, vocab_size: int, params, rng, data):
    """Compute the loss on data wrt params."""
    batch_token_ids = jnp.array(data['transformer/encoder_input:0']) # obs
    batch_labels = jnp.array(data['transformer/decoder_input:0']) # target
    
    logits = transformer(params, rng, data)
    targets = hk.one_hot(batch_labels, vocab_size) # vocab_size = config['n_classes']
    # targets = jax.nn.one_hot(batch_labels, vocab_size) # vocab_size = config['n_classes']
    assert logits.shape == targets.shape
    # mask = jnp.greater(data['obs'], 0)
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits))
    # loss = jnp.sum(loss) / jnp.sum(mask) # TODO: loss * mask
    loss /= targets.shape[0] # TODO: loss * mask
    return loss


class GradientUpdater:
    """A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """
    def __init__(self, net_init, loss_fn,
                 optimizer):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, master_rng, data):
        """Initializes state of the updater."""
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, params, rng, vocab_size, opt_state, data):
        """Updates the state using some data and returns metrics."""
        batch_loss, grads = jax.value_and_grad(self._loss_fn)(params, rng, data)
        updates, opt_state = self._opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, batch_loss


# RUN
def main():
    rng = jax.random.PRNGKey(42)
    data_path = 'natsume.txt'
    n_examples = 25000

    # Create the dataset
    # train_dataset, vocab_size = load(batch_size, sequence_length)
    batch_generator = BatchGenerator()
    batch_generator.load(data_path)
    vocab_size = batch_generator.vocab_size
    train_dataset = batch_generator.get_batch(batch_size=config['batch_size'], n_examples=n_examples)
    test_examples = 100
    test_dataset = batch_generator.get_batch(batch_size=config['batch_size'], n_examples=test_examples, training=False)

    data = next(train_dataset)
    encoder_input = jnp.array(data['transformer/encoder_input:0'])
    decoder_input = jnp.array(data['transformer/decoder_input:0'])

    # Set up the model, loss and updater.
    transformer = hk.transform(transformer_fn(encoder_input, decoder_input)) #, apply_rng=True) # Purify

    loss_fn = functools.partial(lm_loss_fn, transformer.apply, vocab_size)
    params = transformer.init(
        rng,
        data = next(train_dataset),
    )

    total_steps = config['n_epochs'] * (n_examples // config['batch_size'])
    lr_schedule = make_lr_schedule(
        warmup_percentage=0.1, total_steps=total_steps
    )

    # optax is a standalone library renamed from optix
    optimizer = optax.chain(
        optax.clip_by_global_norm(config['max_grad_norm']),
        optax.adam(learning_rate=config['learning_rate']),
        optax.scale_by_schedule(lr_schedule)
    )

    updater = GradientUpdater(params, loss_fn, optimizer)
    opt_state = optimizer.init(params)

    # Initialize parameters.
    logger.info('Initializing parameters...')

    logger.info('Starting train loop...')
    for step, train_batch in enumerate(train_dataset):
        if step % 100 == 0:
            print(f"[Step {step}]")
        if step % 1000 == 0 and step != 0:
            # decoder_target = decoder_input[:, 1:]
            logits = transformer.apply(params, rng, data)
            measure_current_performance(logits, test_dataset, params, rng, n_examples=100, splits='train')

        params, opt_state, batch_loss = updater.update(
            params, rng, vocab_size, opt_state, data
        )

if __name__ == '__main__':
    main()