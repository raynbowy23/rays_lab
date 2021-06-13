import jax
from jax.random import PRNGKey

config = {
    'max_length': 512,
    'embed_dropout_rate': 0.1,
    'fully_connected_drop_rate': 0.1,
    'attention_drop_rate': 0.1,
    'hidden_size': 768,
    'intermediate_size': 3072,
    'n_heads': 12,
    'n_layers': 12,
    'mask_id': 1,
    'weight_stddev': 0.02,

    # For use later in finetuning
    'n_classes': 2,
    'classifier_drop_rate': 0.1,
    'learning_rate': 1e-5,
    'max_grad_norm': 1.0,
    'l2': 0.1,
    'n_epochs': 5,
    'batch_size': 4
}