import collections
import os
import pickle
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = Any  # Works with both FrozenDict and plain dict
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


class Model:
    """Trainable model container — JAX pytree compatible.

    Uses manual pytree registration instead of @flax.struct.dataclass
    for compatibility with JAX 0.4.x through 0.7+ and Flax 0.8 through 0.10+.
    """

    def __init__(self, step: int, apply_fn: nn.Module, params: Params,
                 tx: Optional[optax.GradientTransformation] = None,
                 opt_state: Optional[optax.OptState] = None):
        self.step = step
        self.apply_fn = apply_fn      # NOT a pytree leaf (static)
        self.params = params
        self.tx = tx                  # NOT a pytree leaf (static)
        self.opt_state = opt_state

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        params = variables.get('params', variables)
        if isinstance(params, tuple):
            # Older Flax returns (remaining, value) from pop
            params = params[1]

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def replace(self, **kwargs) -> 'Model':
        return Model(
            step=kwargs.get('step', self.step),
            apply_fn=kwargs.get('apply_fn', self.apply_fn),
            params=kwargs.get('params', self.params),
            tx=kwargs.get('tx', self.tx),
            opt_state=kwargs.get('opt_state', self.opt_state),
        )

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(jax.device_get(self.params), f)

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = pickle.load(f)
        return self.replace(params=params)


# Register Model as a JAX pytree so it works inside jit/grad/vmap
def _model_flatten(model):
    # Children (traced by JAX): step, params, opt_state
    children = (model.step, model.params, model.opt_state)
    # Aux data (static, not traced): apply_fn, tx
    aux = (model.apply_fn, model.tx)
    return children, aux


def _model_unflatten(aux, children):
    step, params, opt_state = children
    apply_fn, tx = aux
    return Model(step=step, apply_fn=apply_fn, params=params,
                 tx=tx, opt_state=opt_state)


jax.tree_util.register_pytree_node(Model, _model_flatten, _model_unflatten)
