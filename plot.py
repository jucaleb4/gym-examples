"""
If not done so, run beforehand:

pip install gymnasium
pip install -e .
"""
import gymnasium as gym
import gym_examples

import numpy as np
import matplotlib.pyplot as plt

def plot_rt_dam_lmp(pnode, season):
    """ 
    Given pnode (must be in gym_examples/envs/oasis_header.csv) and season
    plots RT and DAM LMP.
    """

    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        pnode_id=pnode,
        season=season,
        index_offset=0,
    )


    T = 90*24*4
    rt_lmp_arr = np.zeros(T, dtype=float)
    dam_lmp_arr = np.zeros(T, dtype=float)
    time_arr = np.arange(T)

    _, info = env.reset()
    for t in range(T):
        rt_lmp_arr[t] = info['curr_rt_lmp']
        dam_lmp_arr[t] = info['curr_dam_lmp']
        _, _, _, _, info = env.step(1)

    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    ax.plot(time_arr, rt_lmp_arr, label="RT LMPs", color="red")
    ax.plot(time_arr[24*4:], dam_lmp_arr[:-24*4], label="DAM LMPs", color="black", linestyle="dashed")
    ax.legend()
    ax.set(
        title="LMPs at %s during %s" % (pnode, season),
        xlabel="Elapsed time (15min increments)",
        ylabel="LMP ($/MWh)",
    )
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pnode = "ALAMT3G_7_B1"
    season = 'S23'
    plot_rt_dam_lmp(pnode, season)

