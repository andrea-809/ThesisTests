import gymnasium as gym
import pickle
import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
#from HIQL_actor import get_config, get_taxi_config, HIQLAgent
from FlatIQL_agent import FlatGCIQLAgent as HIQLAgent # drop-in replacement
from FlatIQL_agent import get_taxi_config  # drop-in replacement

DS_PATH = "datasets/taxi_mixed_dataset_trajectories.pkl"

def one_hot_batch(states, num_states=500):
    return jnp.eye(num_states)[np.array(states)]

def sample_batch(dataset, batch_size, num_states=500, subgoal_steps=10, seed=None):
    """
    Sample a batch of transitions for HIQL training.
    
    Goal sampling follows the config probabilities:
      value_goals:      20% current, 50% future trajectory, 30% random
      actor goals:      100% future trajectory
      high_actor_targets: subgoal_steps ahead in the trajectory
    """
    rng = np.random.default_rng(seed)
    trajectories = dataset["trajectories"]

    obs_batch, next_obs_batch = [], []
    actions_batch, rewards_batch, masks_batch = [], [], []
    value_goals_batch = []
    low_actor_goals_batch = []
    high_actor_goals_batch = []
    high_actor_targets_batch = []

    for _ in range(batch_size):
        # ── pick a random trajectory and timestep ──────────────────────────
        traj = trajectories[rng.integers(len(trajectories))]
        T = len(traj["observations"])

        # Need at least one future state, so cap t at T-2
        t = rng.integers(T - 1) if T > 1 else 0

        obs        = traj["observations"][t]
        next_obs   = traj["next_observations"][t]
        action     = traj["actions"][t]
        terminal   = traj["terminals"][t]

        # ── goal-conditioned reward (gc_negative=True → 0 if done else -1) ──
        reward = 0.0 if terminal else -1.0
        mask   = 0.0 if terminal else 1.0

        # ── sample task goal g: future state in same trajectory ───────────
        goal_t = rng.integers(t + 1, T)          # strictly future
        goal   = traj["observations"][goal_t]

        # ── subgoal target w: subgoal_steps ahead, capped at goal_t ───────
        subgoal_t      = min(t + subgoal_steps, goal_t)
        subgoal        = traj["observations"][subgoal_t]

        # ── value goal: 20% current, 50% future traj, 30% random ──────────
        p = rng.random()
        if p < 0.2:
            # current state as goal → V(s, s) ≈ 0, boundary condition
            vg = obs
        elif p < 0.7:
            # future state in same trajectory
            vg = goal
        else:
            # random state from a random trajectory (off-distribution)
            rand_traj = trajectories[rng.integers(len(trajectories))]
            rand_t    = rng.integers(len(rand_traj["observations"]))
            vg        = rand_traj["observations"][rand_t]

        obs_batch.append(obs)
        next_obs_batch.append(next_obs)
        actions_batch.append(action)
        rewards_batch.append(reward)
        masks_batch.append(mask)
        value_goals_batch.append(vg)
        low_actor_goals_batch.append(subgoal)   # low actor: reach the subgoal
        high_actor_goals_batch.append(goal)      # high actor: reach the task goal
        high_actor_targets_batch.append(subgoal) # high actor target: subgoal w

    batch = {
        "observations":       one_hot_batch(obs_batch,              num_states),
        "next_observations":  one_hot_batch(next_obs_batch,         num_states),
        "actions":            jnp.array(actions_batch,              dtype=jnp.int32),
        "rewards":            jnp.array(rewards_batch,              dtype=jnp.float32),
        "masks":              jnp.array(masks_batch,                dtype=jnp.float32),
        "value_goals":        one_hot_batch(value_goals_batch,      num_states),
        "low_actor_goals":    one_hot_batch(low_actor_goals_batch,  num_states),
        "high_actor_goals":   one_hot_batch(high_actor_goals_batch, num_states),
        "high_actor_targets": one_hot_batch(high_actor_targets_batch, num_states),
    }
    return batch

def evaluate(agent, episodes=20):

    returns = []

    for ep in range(episodes):

        state, _ = env.reset()
        done = False
        total_reward = 0
        total_rew_gc = 0

        while not done:

            obs = one_hot_batch(np.array([state]))

            action = int(
                agent.sample_actions(
                    obs,
                    goals=obs,
                    seed=jax.random.PRNGKey(ep)
                )[0]
            )

            next_state, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            state = next_state
            done = terminated or truncated

        returns.append(total_reward)

    return np.mean(returns), returns

# Load offline mixed dataset
with open(DS_PATH, "rb") as f:
    data = pickle.load(f)


NUM_STATES = 500
ex_obs = jnp.zeros((1, NUM_STATES))
ex_actions = jnp.array([5])  # 6 actions in Taxi

# agent = HIQLAgent.create(
#     seed=0,
#     ex_observations=ex_obs,
#     ex_actions=ex_actions,
#     config=get_taxi_config()
# )
agent = HIQLAgent.create(
    seed=0,
    ex_observations=jnp.zeros((1, 500)),
    ex_actions=jnp.array([5]),
    config=get_taxi_config()
)

NUM_TRAIN_STEPS = 30_000
BATCH_SIZE = 128
SUBGOAL_STEPS = 3  

value_losses = []
low_actor_losses = []
high_actor_losses = []
total_losses = []

# ------------------EVALUATION------------------

for step in range(NUM_TRAIN_STEPS):

    batch = sample_batch(data, BATCH_SIZE, subgoal_steps=SUBGOAL_STEPS, seed=step)

    agent, info = agent.update(batch)

    # Store losses
    v_loss = float(info["value/value_loss"])
    l_loss = float(info["low_actor/actor_loss"])
    h_loss = float(info["high_actor/actor_loss"])

    value_losses.append(v_loss)
    low_actor_losses.append(l_loss)
    high_actor_losses.append(h_loss)
    total_losses.append(v_loss + l_loss + h_loss)

    if step % 1000 == 0:
        print(
            f"step {step}",
            "value_loss:", float(info["value/value_loss"]),
            "low_actor_loss:", float(info["low_actor/actor_loss"]),
            "high_actor_loss:", float(info["high_actor/actor_loss"]),
        )

plt.figure(figsize=(10,6))

plt.plot(value_losses, label="Value Loss")
plt.plot(low_actor_losses, label="Low Actor Loss")
plt.plot(high_actor_losses, label="High Actor Loss")
plt.plot(total_losses, label="Total Loss", linewidth=2)

plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("HIQL Training Losses (Taxi Offline)")
plt.legend()
plt.grid()

plt.show(block=True)
# ------------------------------------

env = gym.make("Taxi-v3")

m, ret= evaluate(agent, 100)
print("Mean", m)

plt.figure(figsize=(10,6))

plt.plot(ret, label="Returns")

plt.show(block=True)