"""
This script goes through the finds pnodes in renewable region and TAC with known DAM demand
"""
import numpy as np
import pandas as pd

tac_map_fname    = "20230917_20230918_ATL_TAC_AREA_MAP_N_20230921_10_30_28_v1.csv"
tac_demand_fname = "20230917_20230919_SLD_FCST_DAM_20230920_21_28_35_v1.csv"
renewable_fname  = "20230917_20230918_ATL_PNODE_MAP_N_20230921_11_06_29_v1.csv"

def filter1():
    # TAC pnode pairs
    pnode_to_tac_map = {}
    df = pd.read_csv(tac_map_fname)
    tac_pnode_pair_arr = df[['TAC_AREA_ID', 'PNODE_ID']].to_numpy()

    for (tac, pnode) in tac_pnode_pair_arr:
        # strip TAC_ name
        if 'TAC_' in tac:
            tac_name = tac[len('TAC_'):]
        else:
            print(f"Unexpected TAC name {tac}")
        pnode_to_tac_map[pnode] = tac_name

    # TAC demands
    df = pd.read_csv(tac_demand_fname)
    tac_with_demand_arr = df['TAC_AREA_NAME'].unique()

    pnodes_with_demand = np.array([], dtype=object)
    for (pnode, tac) in pnode_to_tac_map.items():
        print(pnode, tac)
        if tac in tac_with_demand_arr:
            if pnode not in pnodes_with_demand:
                pnodes_with_demand = np.append(pnodes_with_demand, pnode)

    # pnode renewables
    df = pd.read_csv(renewable_fname)
    pnodes_with_renewable = df['PNODE_ID'].unique()

    pnodes_with_demand_and_renewable = np.array([], dtype=object)
    for pnode in pnodes_with_demand:
        if pnode in pnodes_with_renewable:
            pnodes_with_demand_and_renewable = np.append(pnodes_with_demand_and_renewable, pnode)
    print(pnodes_with_demand_and_renewable)

    print("Pnodes of interest:")
    for pnode in pnodes_with_demand_and_renewable:
        print("   pnode:{pnode} TAC: {pnode_to_tac_map[pnode]}")

    # TODO: We do not seem to have pnodes or a reliable way to map pnodes to rewneable regions
    print("Sanity check")
    print(np.sort(pnodes_with_demand))
    print(np.sort(pnodes_with_renewable))

def filter2():
    """ Overlap of pnode (TAC) and renewables """
    df = pd.read_csv(tac_map_fname)
    tac_pnode_pair_arr = df['PNODE_ID'].unique()

    df = pd.read_csv(renewable_fname)
    pnodes_with_renewable = df['PNODE_ID'].unique()

    a = set(tac_pnode_pair_arr).intersection(pnodes_with_renewable)
    print(f"Pnode matchings: {len(a)}. TAC pnodes: {len(tac_pnode_pair_arr)} and Ren pnodes: {len(pnodes_with_renewable)}")

def filter3():
    """ Overlap of pnode (TAC) and demands """
    df = pd.read_csv(tac_map_fname)
    tac_pnode_pair_arr = df['TAC_AREA_ID'].unique()
    tac_names = np.array([], dtype=object)
    for tac in tac_pnode_pair_arr:
        # strip TAC_ name
        if 'TAC_' in tac:
            tac_name = tac[len('TAC_'):]
            tac_names = np.append(tac_names, tac_name)

    df = pd.read_csv(tac_demand_fname)
    tac_demands = df['TAC_AREA_NAME'].unique()

    a = set(tac_names).intersection(tac_demands)
    print(f"TAC matchings: {len(a)}. TAC pnodes: {len(tac_names)} and demands: {len(tac_demands)}")

def filter4():
    # TAC demands
    df = pd.read_csv(tac_demand_fname)
    tac_with_demand_arr = df['TAC_AREA_NAME'].unique()

    # TAC pnode pairs
    df = pd.read_csv(tac_map_fname)
    tac_pnode_pair_arr = df[['TAC_AREA_ID', 'PNODE_ID']].to_numpy()
    pnodes_with_demand = np.array([], dtype=object)
    pnodes_without_demand = np.array([], dtype=object)

    tac_with_demands = np.array([], dtype=object)
    tac_without_demands = np.array([], dtype=object)

    for (tac, pnode) in tac_pnode_pair_arr:
        # strip TAC_ name
        if 'TAC_' in tac:
            tac_name = tac[len('TAC_'):]
            if tac_name in tac_with_demand_arr:
                pnodes_with_demand = np.append(pnodes_with_demand, pnode)
                if tac_name not in tac_with_demands:
                    tac_with_demands = np.append(tac_with_demands, tac_name)
            else:
                pnodes_without_demand = np.append(pnodes_without_demand, pnode)
                if tac_name not in tac_without_demands:
                    tac_without_demands = np.append(tac_without_demands, tac_name)

    # pnode renewables
    df = pd.read_csv(renewable_fname)
    pnodes_with_renewable = df['PNODE_ID'].unique()

    print(f"num pnodes with demand: {len(pnodes_with_demand)}")
    print(f"num pnodes w/o  demand: {len(pnodes_without_demand)}")
    print(f"num pnodes with renews: {len(pnodes_with_renewable)}")
    print(f"dem pnodes: {np.sort(pnodes_with_demand)}")
    print(f"ren pnodes: {np.sort(pnodes_with_renewable)}")
    print(f"num pnodes with both d-r : {len(set(pnodes_with_demand).intersection(pnodes_with_renewable))}")
    print(f"num pnodes with both nd-r: {len(set(pnodes_without_demand).intersection(pnodes_with_renewable))}\n")

    # Based on WECC Balancing Authorities: https://www.wecc.org/Administrative/06-Balancing%20Authority%20Overview.pdf
    print(f"TAC with demands: {np.sort(tac_with_demands)}")

    # Sub-regions: https://haas.berkeley.edu/wp-content/uploads/WP321Appendix.pdf
    print(f"TAC w/o  demands: {np.sort(tac_without_demands)}")

    aa = pnodes_with_demand
    ab = pnodes_with_renewable
    ac = pnodes_without_demand

    ba = np.sort(aa)
    bb = np.sort(ab)
    bc = np.sort(ac)

def filter5():
    """ We have chosen a pnode in LADWP (LA Department of Water and Power) with power in south region """
    # TAC pnode pairs
    df = pd.read_csv(tac_map_fname)
    tac_pnode_pair_arr = df[['TAC_AREA_ID', 'PNODE_ID']].to_numpy()

    ladwp_pnode_arr = []
    south_pnode_arr = []
    south_pnode_ren_arr = []

    for (tac, pnode) in tac_pnode_pair_arr:
        # strip TAC_ name
        if 'TAC_LADWP' == tac:
            ladwp_pnode_arr.append(pnode)
        if 'TAC_SOUTH' == tac:
            south_pnode_arr.append(pnode)

    # pnode renewables
    df = pd.read_csv(renewable_fname)
    ren_pnode_arr = df['PNODE_ID'].unique()
    for pnode in ren_pnode_arr:
        # strip TAC_ name
        if pnode in south_pnode_arr:
            south_pnode_ren_arr.append(pnode)

    print(f"pnode in LADWP: {ladwp_pnode_arr}\n")
    # print(f"pnode in SOUTH: {south_pnode_arr}\n")
    print(f"pnode w/ ren in SOUTH: {south_pnode_ren_arr}\n")

    print(f"Chosen pnode for demand: {ladwp_pnode_arr[3]}")
    print(f"Chosen pnode for renewable: {south_pnode_ren_arr[3]}")

filter5()
