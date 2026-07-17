import jax.numpy as jnp

from flax.struct import dataclass
from jax import Array


@dataclass
class Dataset:
    train_data: tuple[Array, ...]
    train_states: Array
    val_data: tuple[Array, ...]
    val_states: Array
    params: dict[str, Array]

    @property
    def standardised_data(self):
        means = tuple(jnp.mean(d, axis=(0,1), keepdims=True) for d in self.train_data)
        stds = tuple(jnp.std(d, axis=(0,1), keepdims=True) for d in self.train_data)
        scaled_train_data = tuple((d-m)/s for d, m, s in zip(self.train_data, means, stds))
        scaled_val_data = tuple((d-m)/s for d, m ,s in zip(self.val_data, means, stds))

        return Dataset(
            train_data=scaled_train_data,
            train_states=self.train_states,
            val_data=scaled_val_data,
            val_states=self.val_states,
            params=self.params
        )

    @property
    def flatten(self):
        train_shape = self.train_data[0].shape[:2] + (-1,)
        val_shape = self.val_data[0].shape[:2] + (-1,)
        train_data = tuple(jnp.reshape(x, train_shape) for x in self.train_data)
        val_data = tuple(jnp.reshape(x, val_shape) for x in self.val_data)

        return Dataset(
            train_data=train_data,
            train_states=self.train_states,
            val_data=val_data,
            val_states=self.val_states,
            params=self.params
        )

    def __getitem__(self, index):
        """Allow indexing over all J data modalities"""
        return Dataset(
            train_data=tuple(x[index] for x in self.train_data),
            train_states=self.train_states[index],
            val_data=tuple(x[index] for x in self.val_data),     
            val_states=self.val_states[index],
            params=self.params
        )
        
