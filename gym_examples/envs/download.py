import argparse
import numpy as np
import pandas as pd
import requests, zipfile, io
import time
import os
from collections import OrderedDict
from parse import get_fname_from_wildcard, check_pnode_id

def lmp_data_already_exists(node_id, startdate, enddate):
    """ Checks if original LMP or cleaned LMP data exists """
    root = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(root):
        lmp_root_name = os.path.join(root, node_id, "%s_%s" % (startdate, enddate))
        lmp_wildcard_name = "%s_PRC_RTPD_LMP_RTPD_*.csv" % lmp_root_name
        lmp_fname = get_fname_from_wildcard(lmp_wildcard_name, ignore_exception=True)
        already_downloaded = lmp_fname is not None
        if already_downloaded:
            return True

        clean_lmp_wildcard_name = "%s_CLEAN_PRC_RTPD_LMP_RTPD_*.csv" % lmp_root_name
        clean_lmp_fname = get_fname_from_wildcard(clean_lmp_wildcard_name, ignore_exception=True)
        already_downloaded = clean_lmp_fname is not None
        if already_downloaded:
            return True

    return False

def _download(
        node_id: str, 
        item: str, 
        folder:str,
        startdates: list,
        enddates: list,
    ):
    """ 
    Downloads data into designated folder

    :param node_id: which node in CAISO to retrieve LMPs
    :param item: which query item (lmp, demand, renewable)
    :param folder: which folder to extract the files into
    """
    assert os.path.exists(folder), "Must have folder %s before downloading" % folder

    for (startdate, enddate) in zip(startdates, enddates):

        # The time is UTC, which is 8 hours ahead of PST
        startdatetime="%sT08:00-0000" % startdate
        enddatetime="%sT08:00-0000" % enddate

        # for RT LMPs
        if item == "lmp":
            if lmp_data_already_exists(node_id, startdate, enddate):
                print("Skipping download of pnode %s during %s to %s" % (node_id, startdate, enddate))
                continue

            url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_RTPD_LMP&startdatetime={startdatetime}&enddatetime={enddatetime}&version=2&resultformat=6&market_run_id=RTPD&node={node_id}"

        elif item == "demand":
            url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=DAM"

        elif item == "renewable":
            url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_REN_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=DAM"
            # url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_REN_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=RTPD"

        elif item == "actual_renewable":
            """
            Found this by trial and error by looking at this api:
            http://www.caiso.com/Documents/OASIS-InterfaceSpecification_v4_3_5Clean_Spring2017Release.pdf
            """
            url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_REN_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=ACTUAL"

        else:
            raise Exception("Unknown query item %s" % item)

        print("Querying %s..." % url)
        s_time = time.time()
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(folder)

        # ensure at least 11s elapsed to prevent too many queries all at once
        e_time = time.time() - s_time
        sleep_time = 5
        time.sleep(sleep_time)

def _clean(
        node_id: str, 
        startdates: list,
        enddates: list,
    ):
    """
    Takes downloaded LMP data and preserves just the prices in chronological order
    """
    root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(root):
        raise Exception("Cannot find the root path %s" % root)

    for (startdate, enddate) in zip(startdates, enddates):
        lmp_root_name = os.path.join(root, node_id, "%s_%s" % (startdate, enddate))
        lmp_wildcard_name = "%s_PRC_RTPD_LMP_RTPD_*.csv" % lmp_root_name
        lmp_fname = get_fname_from_wildcard(lmp_wildcard_name, ignore_exception=True)

        if lmp_fname is None:
            continue

        df = pd.read_csv(lmp_fname)
    
        lmp_idx = df.index[df['XML_DATA_ITEM'] == 'LMP_PRC'].tolist()
        df = df.iloc[lmp_idx]
        df = df.sort_values(by=['INTERVALSTARTTIME_GMT'])
    
        lmp_arr = np.atleast_2d(df['PRC'].values).T

        output_file = "%s_CLEAN_%s" % (lmp_root_name, lmp_fname[lmp_fname.find("PRC_RTPD_LMP_RTPD"):])
        with open(output_file, "w+") as fp:
            fp.write('lmp\n')
            np.savetxt(fp, lmp_arr, delimiter=",", fmt='%.2f')

        # remove original LMP data since it is large
        os.remove(lmp_fname)

def download(year:int, node_id: str, auxilary: bool):
    """
    Creates download folder and downloads item and downloads data for the full 
    years. See `_download` for more info.

    Example node_id: 'MIL1_3_PASGNODE'

    TODO: How to identify TAC and region?
    TAC and REN: TAC_ECNTR, SP15
    """

    if auxilary:
        items = ['demand', 'renewable', 'actual_renewable']
        folder = "COMMON"
    else:
        items = ['lmp']
        folder = node_id

    # make the folder
    if not os.path.exists(folder):
        os.mkdir(folder)

    days_in_month = OrderedDict([
        (1, 31),
        (2, 29 if year % 4 == 0 else 28),
        (3, 31),
        (4, 30),
        (5, 31),
        (6, 30),
        (7, 31),
        (8, 31),
        (9, 30),
        (10, 31),
        (11, 30),
        (12, 31),
    ])
        
    startdates = []
    enddates = []
    for (mo, num_days) in days_in_month.items():
        if num_days >= 31:
            startdate = "%i%02d01" % (year, mo)
            enddate = "%i%02d31" % (year, mo)
            startdates.append(startdate)
            enddates.append(enddate)

            startdate = "%i%02d31" % (year, mo)
            enddate = "%i%02d01" % (year + 1 if mo == 12 else year, (mo % 12) + 1)
            startdates.append(startdate)
            enddates.append(enddate)

        else:
            startdate = "%i%02d01" % (year, mo)
            enddate = "%i%02d01" % (year, (mo % 12) + 1)
            startdates.append(startdate)
            enddates.append(enddate)

    for item in items:
        _download(node_id, item, folder, startdates, enddates)
    if 'lmp' in items:
        _clean(node_id, startdates, enddates)

    """
    fname = "20230601_20230701_SLD_FCST_DAM_20230927_12_29_00_v1.csv"

    # print(f"Reading file {fname}")
    df = pd.read_csv(fname)

    import ipdb; ipdb.set_trace()
    print(df.head())
    # print(df.describe())
    # print(df.info())
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Download CAISO',
        description='Downloads CAISO data',
    )

    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--node_id', type=str, required=True)
    parser.add_argument('--auxilary', action="store_true")
    args = parser.parse_args()

    check_pnode_id(args.node_id)
    download(args.year, args.node_id, args.auxilary)
