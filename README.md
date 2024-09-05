This is a fork of [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)'s tutorial on building a gym example.
The particular gym example is a simple grid-scale battery energy storage systems (BESS) co-located with photovoltaic (PV).
Real-time and day-ahead location marginal prices (LMPs), solar, and other auxiliary data are downloaded from the CAISO's (California Independent System) [OASIS](https://www.caiso.com/systems-applications/portals-applications/open-access-same-time-information-system-oasis).
To setup the the code, we need to download the data. 

# Data download
`cd` into the directory `gym_examples/envs`. From here, you will download the data via a script. For example, to download real-time LMPs:
``` 
python download.py --year <year> --node_id <node_id>
```

where we used the year 2023 (but you can use others if desired) and `node_id` can be
- COTWDPGE_1_N001
- ALAMT3G_7_B1
- PAULSWT_1_N013
- FREMNT_1_N013

These for pnodes were selected for the experiments in the paper due to their geography and LMP characteristic.
Download times for this can take about 10 minutes due to the size of the datasets.
Download times for the other dataset (i.e., the ones below) are quicker since the files are smaller.

To download day-ahead LMPs,
``` 
python download.py --year 2023 --node_id <node_id> --dam_mkt
```

To download auxiliary data (e.g., solar, demand), 
``` 
python download.py --year 2023 --auxiliary
```
Notice we do not specify the pnode. This is because this auxiliary data is downloaded for regions (not pnodes).
The file `gym_examples/envs/oasis_header.csv` contains custom mappings from some pnodes to the correct regions.

# Setting up BatteryEnv with Gymnasium
Follow Gymnasium's [tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py) for how to install after downloading the dataset.
