# Wildfire
## Description

The wildfire domain simulates a grid-based environment where fires can spread and agents are tasked with extinguishing
them by applying suppressant. The environment is dynamic and partially observable, with fires that can spread across
adjacent tiles and vary in intensity. Fires can also burn out once they reach a certain intensity threshold.

<u>**Environment Dynamics**</u><br>
- Fire Spread: Fires start at designated locations and spread to neighboring tiles, increasing in intensity over
  time. The intensity of the fire influences how much suppressant is needed to extinguish it. Fires will continue
  to spread until they either burn out or are controlled by agents.
- Fire Intensity and Burnout: As fires spread, their intensity increases, making them harder to fight. Once a
  fire reaches a critical intensity, it may burn out naturally, stopping its spread and extinguishing itself.
  However, this is unpredictable, and timely intervention is often necessary to prevent further damage.
- Suppression Mechanism: Agents apply suppressant to the fire to reduce its intensity. However, suppressant is a
  finite resource. When an agent runs out of suppressant, they must leave the environment to refill at a designated
  station before returning to continue fighting fires.

<u>**Environment Openness**</u><br>
- **agent openness**: Environments where agents can dynamically enter and leave, enabling dynamic cooperation and
  multi-agent scenarios with evolving participants.
    - `wildfire`: Agents can run out of suppressant and leave the environment, removing their contributions
      to existing fires. Agents must reason about their collaborators leaving, or new collaborators entering.
- **task openness**: Tasks can be introduced or removed from the environment, allowing for flexbile goal setting
  and adaptable planning models
    - `wildfire`: Fires can spread beyond their original starting point, requiring agents to reason about new
      tasks possibly entering the environment as well as a changing action space: Fires can spread beyond
      their original starting point, requiring agents to reason about new tasks possibly entering the
      environment as well as a changing action space.
- **frame / type openness**: Different frames (e.g. agent abilities or skills) can be added, removed, or modified,
  expending the environmental complexity and requiring agents to infer their neighbors changing abilities.
    - `wildfire`: Agents can damage their equipment over time, and have their capabilities slowly degrade. On
      the other hand, agents might also recieve different equipment upon leaving the environment to resupply.

# Specification

---

| Import             | `from free_range_zoo.envs import wildfire_v0`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Actions            | Discrete & Stochastic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Observations       | Discrete and fully Observed with private observations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Parallel API       | Yes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Manual Control     | No                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Agent Names        | [$firefighter$_0, ..., $firefighter$_n]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| # Agents           | [0, $n_{firefighters}$]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Action Shape       | ($envs$, 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Action Values      | [$fight_0$, ..., $fight_{tasks}$, $noop$ (-1)]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Observation Shape  | TensorDict: { <br>&emsp;**self**: $<ypos, xpos, fire power, suppressant>$<br>&emsp;**others**: $<ypos,xpos,fire power, suppressant>$<br>&emsp;**tasks**: $<y, x, fire level, intensity>$ <br> **batch_size**: $num\_envs$ }                                                                                                                                                                                                                                                                                                                                                                      |
| Observation Values | <u>**self**</u>:<br>&emsp;$ypos$: $[0, max_y]$<br>&emsp;$xpos$: $[0, max_x]$<br>&emsp;$fire\_power\_reduction$: $[0, max_{fire\_power\_reduction}]$<br>&emsp;$suppressant$: $[0, max_{suppressant}]$<br><u>**others**</u>:<br>&emsp;$ypos$: $[0, max_y]$<br>&emsp;$xpos$: $[0, max_x]$<br>&emsp;$fire\_power\_reduction$: $[0, max_{fire\_power\_reduction}]$<br>&emsp;$suppressant$: $[0, max_{suppressant}]$<br> <u>**tasks**</u><br>&emsp;$ypos$: $[0, max_y]$<br>&emsp;$xpos$: $[0, max_x]$<br>&emsp;$fire\_level$: $[0, max_{fire\_level}]$<br>&emsp;$intensity$: $[0, num_{fire\_states}]$ |

---
## Usage
### Parallel API
```python
from free_range_zoo.envs import wildfire_v0

main_logger = logging.getLogger(__name__)

# Initialize and reset environment to initial state
env = wildfire_v0.parallel_env(render_mode="human")
observations, infos = env.reset()

# Initialize agents and give initial observations
agents = []

cumulative_rewards = {agent: 0 for agent in env.agents}

current_step = 0
while not torch.all(env.finished):
    agent_actions = {
        agent_name: torch.stack([agents[agent_name].act()])
        for agent_name in env.agents
    }  # Policy action determination here

    observations, rewards, terminations, truncations, infos = env.step(agent_actions)
    rewards = {agent_name: rewards[agent_name].item() for agent_name in env.agents}

    for agent_name, agent in agents.items():
        agent.observe(observations[agent_name][0])  # Policy observation processing here
        cumulative_rewards[agent_name] += rewards[agent_name]

    main_logger.info(f"Step {current_step}: {rewards}")
    current_step += 1

env.close()
```
### AEC API
```python
from free_range_zoo.envs import wildfire_v0

main_logger = logging.getLogger(__name__)

# Initialize and reset environment to initial state
env = wildfire_v0.parallel_env(render_mode="human")
observations, infos = env.reset()

# Initialize agents and give initial observations
agents = []

cumulative_rewards = {agent: 0 for agent in env.agents}

current_step = 0
while not torch.all(env.finished):
    for agent in env.agent_iter():
        observations, rewards, terminations, truncations, infos = env.last()

        # Policy action determination here
        action = env.action_space(agent).sample()

        env.step(action)

    rewards = {agent: rewards[agent].item() for agent in env.agents}
    cumulative_rewards[agent] += rewards[agent]

    current_step += 1
    main_logger.info(f"Step {current_step}: {rewards}")

env.close()
```

---
