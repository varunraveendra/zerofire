from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
import torch
import pickle

with open('competition_configs/wildfire/WS2.pkl','rb') as f:
    wildfire_configuration = pickle.load(f)

env = wildfire_v0.parallel_env(
    # render_mode="human",
    max_steps = 100,
    parallel_envs = 1,
    configuration = wildfire_configuration,
    device=torch.device('cpu'),
    log_directory = "test_logging",
    override_initialization_check=True
)
env.reset()
env = action_mapping_wrapper_v0(env)
observations, infos = env.reset()

from free_range_zoo.envs.wildfire.baselines import NoopBaseline, RandomBaseline, StrongestBaseline, WeakestBaseline

agents = {
    env.agents[0]: StrongestBaseline(agent_name = "firefighter_1", parallel_envs = 1),
    env.agents[1]: WeakestBaseline(agent_name = "firefighter_2", parallel_envs = 1),
    env.agents[2]: WeakestBaseline(agent_name = "firefighter_3", parallel_envs = 1)
}

print(len(env.agents))
while not torch.all(env.finished):
    for agent_name, agent in agents.items():
        # print(observations[agent_name])
        agent.observe(observations[agent_name])  # Policy observation 


    agent_actions = {
            agent_name:agents[agent_name].act(action_space = env.action_space(agent_name))
        for agent_name in env.agents
    }  # Policy action determination here
    print(agent_actions)
    observations, rewards, terminations, truncations, infos = env.step(agent_actions)
    

env.close()



import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_path = "test_logging/0.csv"  
df = pd.read_csv(csv_path)

# Convert rewards to numeric (in case of NULLs or strings)
for col in ["firefighter_1_rewards", "firefighter_2_rewards", "firefighter_3_rewards"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df["step"], df["firefighter_1_rewards"], label="Firefighter rewards")

plt.xlabel("Simulation Step")
plt.ylabel("Reward")
plt.title("Firefighter Rewards Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
