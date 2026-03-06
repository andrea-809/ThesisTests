from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize

from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize


class HIQLAgent(flax.struct.PyTreeNode):
    """Flat goal-conditioned IQL agent. Drop-in replacement for HIQLAgent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        # Use concatenated [obs; goal] as input to value network
        obs_goal      = jnp.concatenate([batch['observations'],      batch['value_goals']], axis=-1)
        next_obs_goal = jnp.concatenate([batch['next_observations'], batch['value_goals']], axis=-1)

        (next_v1_t, next_v2_t) = self.network.select('target_value')(next_obs_goal)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(obs_goal)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(obs_goal, params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def low_actor_loss(self, batch, grad_params):
        obs_goal      = jnp.concatenate([batch['observations'],      batch['low_actor_goals']], axis=-1)
        next_obs_goal = jnp.concatenate([batch['next_observations'], batch['low_actor_goals']], axis=-1)

        (v1, v2)   = self.network.select('value')(obs_goal)
        (nv1, nv2) = self.network.select('value')(next_obs_goal)
        v  = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        # Normalize advantages for stability
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('low_actor')(obs_goal, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }

    def high_actor_loss(self, batch, grad_params):
        # Stub — keeps interface identical to HIQLAgent
        return jnp.array(0.0), {
            'actor_loss': jnp.array(0.0),
            'adv': jnp.array(0.0),
            'bc_log_prob': jnp.array(0.0),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = value_loss + low_actor_loss  # high actor excluded
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        obs_goal = jnp.concatenate([observations, goals], axis=-1)
        dist = self.network.select('low_actor')(obs_goal, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Pre-concatenate [obs; goal] — input dim is 2 * obs_dim
        ex_obs_goal = jnp.concatenate([ex_observations, ex_observations], axis=-1)

        # Value networks take flat [obs; goal] vector directly, no GCEncoder
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=None,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=None,
        )

        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=None,
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=None,
            )

        network_info = dict(
            value=(value_def,               ex_obs_goal),
            target_value=(target_value_def, ex_obs_goal),
            low_actor=(low_actor_def,        ex_obs_goal),
        )
        networks     = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def    = ModuleDict(networks)
        network_tx     = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network        = TrainState.create(network_def, network_params, tx=network_tx)

        network.params['modules_target_value'] = network.params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    # Unchanged
    ...

def get_taxi_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='flat_gciql_taxi',
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(128, 128),
            value_hidden_dims=(128, 128),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.7,
            low_alpha=3.0,
            high_alpha=3.0,      # unused but kept for interface compatibility
            subgoal_steps=4,     # unused but kept for interface compatibility
            rep_dim=32,          # unused but kept for interface compatibility
            low_actor_rep_grad=False,
            const_std=True,
            discrete=True,
            encoder=None,
            dataset_class='HGCDataset',
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=1,
        )
    )
    return config