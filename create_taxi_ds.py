import gymnasium as gym
import numpy as np
import pickle
import random

folder = 'datasets/'

n_episodes = 10000
max_steps = 200

# Use Taxi-v3 environment and unwrapped to access the transition table
env = gym.make("Taxi-v3", render_mode=None).unwrapped

n_states = env.observation_space.n
n_actions = env.action_space.n

# ----------------------
# Step 1: Compute Optimal Policy via Value Iteration
# ----------------------
V = np.zeros(n_states)
gamma = 0.99
theta = 1e-6

while True:
    delta = 0
    for s in range(n_states):
        v = V[s]
        q_sa = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_s, reward, done in env.P[s][a]:
                q_sa[a] += prob * (reward + gamma * V[next_s])
        V[s] = np.max(q_sa)
        delta = max(delta, abs(v - V[s]))
    if delta < theta:
        break

# Extract optimal policy
policy = np.zeros(n_states, dtype=int)
for s in range(n_states):
    q_sa = np.zeros(n_actions)
    for a in range(n_actions):
        for prob, next_s, reward, done in env.P[s][a]:
            q_sa[a] += prob * (reward + gamma * V[next_s])
    policy[s] = np.argmax(q_sa)

print("Optimal policy computed.")

dataset = []

for ep in range(n_episodes):
    state, _ = env.reset()
    done = False
    for t in range(max_steps):
        action = policy[state]
        next_state, reward, terminated, truncated, info = env.step(action)
        done_flag = terminated or truncated
        
        dataset.append((state, action, reward, next_state, done_flag))
        state = next_state
        if done_flag:
            break

with open(folder+"taxi_expert_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print(f"Saved {len(dataset)} transitions from {n_episodes} episodes.")

dataset_mixed = []

for ep in range(n_episodes):
    state, _ = env.reset()
    done = False
    for t in range(max_steps):
        # Choose policy for this step: 70% Expert, 30% Random
        if random.random() < 0.7:
            action = policy[state]  # Expert action
        else:
            action = env.action_space.sample()  # Random action

        next_state, _, terminated, truncated, info = env.step(action)
        done_flag = terminated or truncated

        dataset_mixed.append((state, action, reward, next_state, done_flag))
        state = next_state
        if done_flag:
            break

print(f"Generated {len(dataset_mixed)} transitions from {n_episodes} episodes.")

obs = []
actions = []
rewards = []
next_obs = []
dones = []

for s, a, r, s_, d in dataset_mixed:
    obs.append(s)
    actions.append(a)
    rewards.append(r)
    next_obs.append(s_)
    dones.append(d)

offline_mixed_dataset = {
    'observations': np.array(obs),
    'actions': np.array(actions),
    'rewards': np.array(rewards),
    'next_observations': np.array(next_obs),
    'terminals': np.array(dones)
}

with open(folder+"taxi_mixed_dataset_offline.pkl", "wb") as f:
    pickle.dump(offline_mixed_dataset, f)

print("Mixed offline dataset saved in standard format.")

trajectories = []
current_traj = {
    "observations": [],
    "actions": [],
    "rewards": [],
    "next_observations": [],
    "terminals": []
}

for s, a, r, s_, d in dataset_mixed:

    current_traj["observations"].append(s)
    current_traj["actions"].append(a)
    current_traj["rewards"].append(r)
    current_traj["next_observations"].append(s_)
    current_traj["terminals"].append(d)

    # If episode finished → store trajectory
    if d:
        trajectories.append({
            "observations": np.array(current_traj["observations"]),
            "actions": np.array(current_traj["actions"]),
            "rewards": np.array(current_traj["rewards"]),
            "next_observations": np.array(current_traj["next_observations"]),
            "terminals": np.array(current_traj["terminals"])
        })

        # Reset trajectory
        current_traj = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "terminals": []
        }

offline_mixed_dataset = {
    "trajectories": trajectories
}

with open(folder+"taxi_mixed_dataset_trajectories.pkl", "wb") as f:
    pickle.dump(offline_mixed_dataset, f)

print(f"Saved {len(trajectories)} trajectories.")