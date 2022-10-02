import jax
import jax.numpy as jnp


class Sarsa:
    def __init__(self, shape, learning_rate, discount_factor, epsilon, seed=42):
        assert (
            0 < learning_rate < 1
        ), f"Learning rate must be between 0 and 1 and not {learning_rate}"
        assert (
            0 < discount_factor < 1
        ), f"Discount factor must be between 0 and 1 and not {discount_factor}"

        # Assuming that the last dimension is the action space
        self.action_space = shape[-1]

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon

        self.q_table = jnp.zeros(size=shape)

        self.key = jax.random.PRNGKey(seed)

    def update(self, s_t, a_t, reward, s_next, a_next):
        self.q_table[tuple(s_t)][
            a_t
        ] += self.learning_rate * self._get_temporal_difference(
            s_t, a_t, reward, s_next, a_next
        )

    def _get_temporal_difference(self, s_t, a_t, reward, s_next, a_next):
        return (
            self._get_temporal_target(reward, s_next, a_next)
            - self.q_table[tuple(s_t)][a_t]
        )

    def _get_temporal_target(self, reward, s_next, a_next):
        return reward + self.discount_factor * self.q_table[tuple(s_next)][a_next]

    def get_best_action(self, s_t):
        return jnp.argmax(self.q_table[tuple(s_t)]).item()

    def get_action(self, s_t):
        self.key, *subkeys = jax.random.split(self.key, 2)
        uniform = jnp.random.uniform(subkeys[0])
        if uniform < self.epsilon:
            return jnp.random.randint(subkeys[1], self.action_space)
        else:
            return self.get_best_action(s_t)
