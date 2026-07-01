# from gym.envs.registration import register
from gymnasium.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
    max_episode_steps=1000,
)

register(
    id="gym_examples/GridWorld-v1",
    entry_point="gym_examples.envs:GridWorldModelEnv",
    max_episode_steps=int(1e9),
)

register(
    id="gym_examples/BatteryEnv-v0",
    entry_point="gym_examples.envs:SimpleBatteryEnv",
    max_episode_steps=1000,
)

register(
    id="gym_examples/SimpleWorld-v0",
    entry_point="gym_examples.envs:SimpleWorldEnv",
    max_episode_steps=1000,
)

register(
    id="gym_examples/LQREnv-v0",
    entry_point="gym_examples.envs:LQREnv",
    max_episode_steps=1000,
)

register(
    id="gym_examples/GiniPortfolioEnv-v0",
    entry_point="gym_examples.envs:GiniPortfolioEnv",
    max_episode_steps=1000,
)

register(
    id="gym_examples/InventoryEnv-v0",
    entry_point="gym_examples.envs:InventoryEnv",
    max_episode_steps=32,
)
