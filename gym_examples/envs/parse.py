"""
This file parses through the files to gather data.
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_fname_from_wildcard(wildcard_fname: str):
    fnames = glob.glob(wildcard_fname)
    if len(fnames) != 1:
        raise Exception("Expected one file for wildcard %s, got %i" % (
            wildcard_fname, 
            len(fnames)
        ))
    return fnames[0]

def get_caiso_data(node_id, print_data=False):
    """
    Parses and returns data for ____ during 06/30/2023-08/30/2023

    :return lmp_arr: RT (every 15m) LMP (unit: $/MWh)
    :return demand_arr: DA (every 1hr) demand power (unit: MW)
    :return solar_arr: DA (every 1hr) solar power forecast (unit: MW)
    :return wind_arr: DA (every 1hr) wind power forecast (unit: MW)
    :return actual_solar_arr: DA (every 1 hr) actual solar (unit: MW)
    """
 
    # root = "/storage/home/hcoda1/9/cju33/gym-examples/gym_examples/envs"
    # root = "/Users/calebju/Code/github/gym-examples/gym_examples/envs"
    # TODO: The location of `Github` needs to be changed on ad-hoc basis
    # root = os.path.join(os.path.expanduser("~"), "gym-examples", "gym_examples", "envs", node_id)
    root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(root):
        raise Exception("Cannot find the root path %s" % root)
        # root = os.path.join(os.path.expanduser("~"), "Code", "github", "gym-examples/gym_examples/envs")

    lmp_arr = np.array([], dtype=float)
    demand_arr = np.array([], dtype=float)
    solar_arr = np.array([], dtype=float)
    wind_arr = np.array([], dtype=float)
    actual_solar_arr = np.array([], dtype=float)

    if not os.path.isfile(os.path.join(root, "oasis_header.csv")):
        raise Exception("Missing 'oasis_header.csv' file")

    df = pd.read_csv(os.path.join(root, "oasis_header.csv"), header="infer")
    if np.sum(df["pnode"] == node_id) == 0:
        raise Exception("Pnode %s does not exist in 'oasis_header.csv' directory" % node_id)
    tac_id = df[df["pnode"] == node_id]["tac"].iloc[0]
    zone_id = df[df["pnode"] == node_id]["zone"].iloc[0]

    if not os.path.exists(os.path.join(root, node_id)):
        raise Exception("Folder %s does not exist" % os.path.join(root, node_id))

    startdates = ["601", "701", "731", "801", "831"]
    enddates = ["701", "731", "801", "831", "901"]

    for (startdate, enddate) in zip(startdates, enddates):

        startdate=f"20230{startdate}"
        enddate=f"20230{enddate}"

        lmp_root_name = os.path.join(root, node_id, "%s_%s" % (startdate, enddate))
        root_name = os.path.join(root, "COMMON", "%s_%s" % (startdate, enddate))
        lmp_wildcard_name = "%s_PRC_RTPD_LMP_RTPD_*.csv" % lmp_root_name
        demand_wildcard_name = "%s_SLD_FCST_DAM_*.csv" % root_name
        renew_wildcard_name = "%s_SLD_REN_FCST_DAM_*.csv" % root_name
        actual_renew_wildcard_name = "%s_SLD_REN_FCST_ACTUAL_*.csv" % root_name

        lmp_fname = get_fname_from_wildcard(lmp_wildcard_name)
        demand_fname = get_fname_from_wildcard(demand_wildcard_name)
        renew_fname = get_fname_from_wildcard(renew_wildcard_name)
        actual_renew_fname = get_fname_from_wildcard(actual_renew_wildcard_name)

        """
    for fnames in zip(lmp_fnames, demand_fnames, renew_fnames, actual_renew_fnames):
        (lmp_fname, demand_fname, renew_fname, actual_renew_fname) = fnames
        """
    
        # extract LMP @ MIL1_3_PASGNODE: only care about LMPs right now and sort by datetime
        df = pd.read_csv(lmp_fname)
    
        lmp_idx = df.index[df['XML_DATA_ITEM'] == 'LMP_PRC'].tolist()
        df = df.iloc[lmp_idx]
        df = df.sort_values(by=['INTERVALSTARTTIME_GMT'])
    
        lmp_arr = np.append(lmp_arr, df['PRC'].values)
    
        # extract demand at TAC LADWP
        df = pd.read_csv(demand_fname)
    
        dem_idx = df.index[df['TAC_AREA_NAME'] == tac_id].tolist()
        df = df.iloc[dem_idx]
        df = df.sort_values(by=['INTERVALSTARTTIME_GMT'])
    
        demand_arr = np.append(demand_arr, df['MW'].values)
    
        # extract renewable at SP15
        df = pd.read_csv(renew_fname)
    
        sol_idx = df.index[(df['RENEWABLE_TYPE'] == "Solar") & (df['TRADING_HUB'] == zone_id)].tolist()
        wnd_idx = df.index[(df['RENEWABLE_TYPE'] == "Wind") & (df['TRADING_HUB'] == zone_id)].tolist()
    
        df_sol = df.iloc[sol_idx]
        df_sol = df_sol.sort_values(by=['INTERVALSTARTTIME_GMT'])
        df_wnd = df.iloc[wnd_idx]
        df_wnd = df_wnd.sort_values(by=['INTERVALSTARTTIME_GMT'])
    
        solar_arr = np.append(solar_arr, df_sol['MW'].values)
        wind_arr = np.append(wind_arr, df_wnd['MW'].values)

        # extract actual rewnewable at SP15
        df = pd.read_csv(actual_renew_fname)
    
        sol_idx = df.index[(df['RENEWABLE_TYPE'] == "Solar") & (df['TRADING_HUB'] == zone_id)].tolist()
        wnd_idx = df.index[(df['RENEWABLE_TYPE'] == "Wind") & (df['TRADING_HUB'] == zone_id)].tolist()
    
        df_sol = df.iloc[sol_idx]
        df_sol = df_sol.sort_values(by=['INTERVALSTARTTIME_GMT'])
    
        # TODO: Why is the length 719 rather than 720=24*30?
        actual_solar_arr = np.append(actual_solar_arr, df_sol['MW'].values)
    
        if print_data:
            print(f"#lmps={len(lmp_arr)} #demands={len(demand_arr)} #solar={len(solar_arr)} #winds={len(wind_arr)}")
    
    # simple data analysis
    if print_data:
        print(f"Extrema LMP   : {np.min(lmp_arr)}, {np.mean(lmp_arr)}, {np.max(lmp_arr)}")
        print(f"Extrema demand: {np.min(demand_arr)}, {np.mean(demand_arr)}, {np.max(demand_arr)}")
        print(f"Extrema wind  : {np.min(wind_arr)}, {np.mean(wind_arr)}, {np.max(wind_arr)}")
        print(f"Extrema solar : {np.min(solar_arr)}, {np.mean(solar_arr)}, {np.max(solar_arr)}")

    return (lmp_arr, demand_arr, solar_arr, wind_arr, actual_solar_arr)

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
    get_caiso_data('FREMNT_1_N013', print_data=False)
    # simple_visualization()
