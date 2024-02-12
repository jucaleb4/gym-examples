"""
This file parses through the files to gather data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_caiso_data(print_data=False):
 
    root = "/storage/home/hcoda1/9/cju33/gym-examples/gym_examples/envs"
    # TODO: Find a way to make this automated
    root = "/Users/calebju/Code/github/gym-examples/gym_examples/envs"

    lmp_fnames = ["20230601_20230701_PRC_RTPD_LMP_RTPD_20230927_12_19_12_v2.csv", "20230701_20230731_PRC_RTPD_LMP_RTPD_20230927_12_20_50_v2.csv", "20230731_20230830_PRC_RTPD_LMP_RTPD_20230927_12_23_20_v2.csv"]
    demand_fnames = ["20230601_20230701_SLD_FCST_DAM_20230927_12_29_00_v1.csv", "20230701_20230731_SLD_FCST_DAM_20230927_12_29_21_v1.csv", "20230731_20230830_SLD_FCST_DAM_20230927_12_29_39_v1.csv"]
    renew_fnames = ["20230601_20230701_SLD_REN_FCST_DAM_20230927_12_31_40_v1.csv", "20230701_20230731_SLD_REN_FCST_DAM_20230927_12_31_45_v1.csv", "20230731_20230830_SLD_REN_FCST_DAM_20230927_12_31_50_v1.csv"]
    
    lmp_arr = np.array([], dtype=float)
    demand_arr = np.array([], dtype=float)
    solar_arr = np.array([], dtype=float)
    wind_arr = np.array([], dtype=float)
    
    for (lmp_fname, demand_fname, renew_fname) in zip(lmp_fnames, demand_fnames, renew_fnames):
    
        # extract LMP @ MIL1_3_PASGNODE: only care about LMPs right now and sort by datetime
        df = pd.read_csv(f"{root}/{lmp_fname}")
    
        lmp_idx = df.index[df['XML_DATA_ITEM'] == 'LMP_PRC'].tolist()
        df = df.iloc[lmp_idx]
        df = df.sort_values(by=['INTERVALSTARTTIME_GMT'])
    
        lmp_arr = np.append(lmp_arr, df['PRC'].values)
    
        # print(df.info(), "\n")
        # print(df.head(), "\n")
    
        # extract demand at TAC LADWP
        df = pd.read_csv(f"{root}/{demand_fname}")
    
        dem_idx = df.index[df['TAC_AREA_NAME'] == "LADWP"].tolist()
        df = df.iloc[dem_idx]
        df = df.sort_values(by=['INTERVALSTARTTIME_GMT'])
    
        demand_arr = np.append(demand_arr, df['MW'].values)
    
        # extract renewable at SP15
        df = pd.read_csv(f"{root}/{renew_fname}")
    
        sol_idx = df.index[(df['RENEWABLE_TYPE'] == "Solar") & (df['TRADING_HUB'] == "SP15")].tolist()
        wnd_idx = df.index[(df['RENEWABLE_TYPE'] == "Wind") & (df['TRADING_HUB'] == "SP15")].tolist()
    
        df_sol = df.iloc[sol_idx]
        df_sol = df_sol.sort_values(by=['INTERVALSTARTTIME_GMT'])
        df_wnd = df.iloc[wnd_idx]
        df_wnd = df_wnd.sort_values(by=['INTERVALSTARTTIME_GMT'])
    
        solar_arr = np.append(solar_arr, df_sol['MW'].values)
        wind_arr = np.append(wind_arr, df_wnd['MW'].values)
    
        if print_data:
            print(f"#lmps={len(lmp_arr)} #demands={len(demand_arr)} #solar={len(solar_arr)} #winds={len(wind_arr)}")
    
    # simple data analysis
    if print_data:
        print(f"Extrema LMP   : {np.min(lmp_arr)}, {np.mean(lmp_arr)}, {np.max(lmp_arr)}")
        print(f"Extrema demand: {np.min(demand_arr)}, {np.mean(demand_arr)}, {np.max(demand_arr)}")
        print(f"Extrema wind  : {np.min(wind_arr)}, {np.mean(wind_arr)}, {np.max(wind_arr)}")
        print(f"Extrema solar : {np.min(solar_arr)}, {np.mean(solar_arr)}, {np.max(solar_arr)}")
    
    return (lmp_arr, demand_arr, solar_arr, wind_arr)

def simple_visualization():
    lmp_arr, demand_arr, solar_arr, wind_arr = get_caiso_data()

    lmp_hourly_arr = np.convolve(lmp_arr, (1./4)*np.ones(4), mode='valid')[::4]

    # look at two weeks
    num_hours = 24 * 90
    lmp_hourly_arr = lmp_hourly_arr[0:num_hours]
    demand_arr = demand_arr[0:num_hours]
    solar_arr = solar_arr[0:num_hours]
    wind_arr = wind_arr[0:num_hours]

    rl_actions = np.load("battery_actions_092823.npy")
    # shift
    rl_actions = 25 * (rl_actions-1)

    assert len(lmp_hourly_arr) == len(demand_arr) == len(solar_arr) == len(wind_arr) 
    xs = np.arange(len(lmp_hourly_arr))

    plt.style.use('ggplot')

    # plot just CAISO
    if False:
        fig, axes = plt.subplots(nrows=2, ncols=2)

        axes[0,0].plot(xs, lmp_hourly_arr, color="black")
        axes[0,0].set(title="Hourly LMP price ($)")

        axes[0,1].plot(xs, demand_arr, color="red")
        axes[0,1].set(title="Hourly Demand (MW)")

        axes[1,0].plot(xs, solar_arr, color="yellow")
        axes[1,0].set(title="Hourly Solar (MW)")

        axes[1,1].plot(xs, wind_arr, color="green")
        axes[1,1].set(title="Hourly Wind (MW)")
    else:
        fig, axes = plt.subplots(nrows=2)
        num_itervals = 4 * 24 * 7
        lmp_arr = lmp_arr[:num_itervals]
        rl_actions = rl_actions[:num_itervals]
        xs = np.arange(len(lmp_arr))

        axes[0].plot(xs, lmp_arr, color="black")
        axes[0].set(title="RT LMP price ($)")

        axes[1].plot(xs, rl_actions, 'g.')
        axes[1].set(title="Battery energy transfer (MW)")

    plt.show()

if __name__ == '__main__':
    # get_caiso_data(print_data=True)
    simple_visualization()
