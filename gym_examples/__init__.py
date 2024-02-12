# from gym.envs.registration import register
from gymnasium.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
    max_episode_steps=1000,
)

register(
    id="gym_examples/BatteryEnv-v0",
    entry_point="gym_examples.envs:SimpleBatteryEnv",
    max_episode_steps=1000,
)
