import argparse
import numpy as np
import pandas as pd
import requests, zipfile, io
import time
import os
from datetime import datetime
from collections import OrderedDict
from parse import get_fname_from_wildcard, check_pnode_id

def lmp_data_already_exists(node_id, startdate, enddate, is_rt_data):
    """ Checks if original LMP or cleaned LMP data exists """
    root = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(root):
        lmp_root_name = os.path.join(root, node_id, "%s_%s" % (startdate, enddate))
        if is_rt_data:
            lmp_wildcard_name = "%s_PRC_RTPD_LMP_RTPD_*.csv" % lmp_root_name
        else:
            lmp_wildcard_name = "%s_PRC_LMP_DAM_*.csv" % lmp_root_name
        lmp_fname = get_fname_from_wildcard(lmp_wildcard_name, ignore_exception=True)
        already_downloaded = lmp_fname is not None
        if already_downloaded:
            return True

        if is_rt_data:
            clean_lmp_wildcard_name = "%s_CLEAN_PRC_RTPD_LMP_RTPD_*.csv" % lmp_root_name
        else:
            clean_lmp_wildcard_name = "%s_CLEAN_PRC_LMP_DAM_*.csv" % lmp_root_name
        clean_lmp_fname = get_fname_from_wildcard(clean_lmp_wildcard_name, ignore_exception=True)
        already_downloaded = clean_lmp_fname is not None
        if already_downloaded:
            return True

    return False

def auxiliary_data_already_exists(datatype, startdate, enddate):
    """ Checks if original LMP or cleaned LMP data exists """
    root = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(root):
        dates = '%s_%s' % (startdate, enddate)
        data_wildcard_name = "%s_SLD_%s_*.csv" % (dates, datatype)
        data_fname = get_fname_from_wildcard(data_wildcard_name, ignore_exception=True)
        already_downloaded = data_fname is not None
        if already_downloaded:
            return True

    return False

def _download(
        node_id: str, 
        item: str, 
        folder:str,
        startdates: list,
        enddates: list,
        is_rt_data: bool=True,
    ):
    """ 
    Downloads data into designated folder. 

    Here is the typical url (see 2017 API for examples):

      RT: http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_RTPD_LMP&startdatetime=20231101T08:00-0000&enddatetime=20231201T08:00-0000&version=2&resultformat=6&market_run_id=RTPD&node=ALAMT3G_7_N002
      DAM: http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_LMP&startdatetime=20231101T08:00-0000&enddatetime=20231201T08:00-0000&version=1&resultformat=6&market_run_id=DAM&node=ALAMT3G_7_N002

    :param node_id: which node in CAISO to retrieve LMPs
    :param item: which query item (lmp, demand, renewable)
    :param folder: which folder to extract the files into
    :params startdates, enddates: array of start ane end dates
    :param is_rt_data: boolean if we want RT data. If not, we will get DAM
    """
    assert os.path.exists(folder), "Must have folder %s before downloading" % folder

    for (startdate, enddate) in zip(startdates, enddates):

        # The time is UTC, which is 8 hours ahead of PST
        startdatetime="%sT08:00-0000" % startdate
        enddatetime="%sT08:00-0000" % enddate

        # for RT LMPs
        if item == "lmp":
            assert node_id is not None
            if lmp_data_already_exists(node_id, startdate, enddate, is_rt_data):
                print("Data for pnode %s during %s to %s already exists. Skipping" % (node_id, startdate, enddate))
                continue

            if is_rt_data:
                url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_RTPD_LMP&startdatetime={startdatetime}&enddatetime={enddatetime}&version=2&resultformat=6&market_run_id=RTPD&node={node_id}"
            else:
                url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_LMP&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=DAM&node={node_id}"

        elif item == "demand":
            if auxiliary_data_already_exists('FCST_DAM', startdate, enddate):
                print("Skipping download of FCST DAM during %s to %s" % (startdate, enddate))
                continue

            url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=DAM"

        elif item == "renewable":
            if auxiliary_data_already_exists('REN_FCST_DAM', startdate, enddate):
                print("Skipping download of FCST DAM during %s to %s" % (startdate, enddate))
                continue

            url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_REN_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=DAM"
            # url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_REN_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=RTPD"

        elif item == "actual_demand":
            if auxiliary_data_already_exists('FCST_DAM_ACTUAL', startdate, enddate):
                print("Skipping download of FCST DAM during %s to %s" % (startdate, enddate))
                continue

            url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=ACTUAL"

        elif item == "actual_renewable":
            """
            Found this by trial and error by looking at this api:
            http://www.caiso.com/Documents/OASIS-InterfaceSpecification_v4_3_5Clean_Spring2017Release.pdf
            """
            if auxiliary_data_already_exists('REN_FCST_ACTUAL', startdate, enddate):
                print("Skipping download of FCST DAM during %s to %s" % (startdate, enddate))
                continue

            url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_REN_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=ACTUAL"

        else:
            raise Exception("Unknown query item %s" % item)

        print("Querying %s." % url, end=' ')
        s_time = time.time()
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(folder)

        # elapse >=1sec to avoid many queries at once
        e_time = time.time() - s_time
        sleep_time = max(5-e_time, 0) + 1
        time.sleep(sleep_time)
        print("Complete (time %.2fs)" % (time.time() - s_time))

def check_correct_length(n, startdate, enddate, n_in_one_day):
    """
    Checks whether number of objects is correct. 

    :param n: number of data points from downloaded data
    :params startdate, enddate: start and enddate (as string in YYYYMMDD format)
    :param n_in_on_day: how many data points we expect in a day (e.g., for 15m increments, expect 4*24=96)
    :return correct: whether number of data points matches what is expected
    """
    date_format = "%Y%m%d"
    startdate_obj = datetime.strptime(startdate, date_format).date()
    enddate_obj   = datetime.strptime(enddate, date_format).date()
    n_days = (enddate_obj - startdate_obj).days

    expected_n = n_days * n_in_one_day

    if expected_n != n:
        warnings.warn("Recieved %d (expected %d) for data between %s to %s" % (
            n,
            expected_n,
            startdate,
            enddate,
        ))

        return False

    return True

def is_missing_data(df, fname):
    startdt_str_arr = df["INTERVALSTARTTIME_GMT"].to_numpy()
    enddt_str_arr = df["INTERVALENDTIME_GMT"].to_numpy()

    missing_data = False

    for t in range(len(startdt_str_arr)-1):
        if startdt_str_arr[t+1] != enddt_str_arr[t]:
            print("File %s missing data from %s to %s" % (
                fname,
                enddt_str_arr[t],
                startdt_str_arr[t],
            ))
            missing_data = True

    return missing_data

def _clean(
        node_id: str, 
        startdates: list,
        enddates: list,
        is_rt_data: bool=True,
    ):
    """
    Takes downloaded LMP data (downloaded from CAISO), extracts prices, and
    sorts in chronological order. We do data extraction to reduce the datasize,
    since the LMP data may contain extraneous data.

    TODO: Do we need to do data imputation?
    """
    root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(root):
        raise Exception("Cannot find the root path %s" % root)

    for (startdate, enddate) in zip(startdates, enddates):
        lmp_root_name = os.path.join(root, node_id, "%s_%s" % (startdate, enddate))
        if is_rt_data: 
            lmp_wildcard_name = "%s_PRC_RTPD_LMP_RTPD_*.csv" % lmp_root_name
        else:
            lmp_wildcard_name = "%s_PRC_LMP_DAM_*.csv" % lmp_root_name
        lmp_fname = get_fname_from_wildcard(lmp_wildcard_name, ignore_exception=True)

        if lmp_fname is None:
            print("Missing file %s for cleaning" % lmp_wildcard_name.split("/")[-1])
            continue

        df = pd.read_csv(lmp_fname)
    
        lmp_idx = df.index[df['XML_DATA_ITEM'] == 'LMP_PRC'].tolist()
        df = df.iloc[lmp_idx]
        df = df.sort_values(by=['INTERVALSTARTTIME_GMT'])

        if is_missing_data(df, lmp_fname):
            print("File %s has missing data. Not deleting" % lmp_fname[lmp_fname.find("PRC_RTPD_LMP_RTPD"):])
            continue
    
        # DA lists data under column 'MW', while RT lists under 'PRC'
        if is_rt_data: 
            lmp_arr = np.atleast_2d(df['PRC'].values).T
        else:
            lmp_arr = np.atleast_2d(df['MW'].values).T

        n_data_in_day = 4*24 if is_rt_data else 24
        has_correct_length = check_correct_length(len(lmp_arr), startdate, enddate, n_data_in_day)

        if not has_correct_length:
            print("Data %s has incorrect data points. Not deleting" % lmp_fname[lmp_fname.find("PRC_RTPD_LMP_RTPD"):])
            continue

        if is_rt_data:
            output_file = "%s_CLEAN_%s" % (lmp_root_name, lmp_fname[lmp_fname.find("PRC_RTPD_LMP_RTPD"):])
        else:
            output_file = "%s_CLEAN_%s" % (lmp_root_name, lmp_fname[lmp_fname.find("PRC_LMP_DAM"):])
        with open(output_file, "w+") as fp:
            fp.write('lmp\n')
            np.savetxt(fp, lmp_arr, delimiter=",", fmt='%.2f')

        # remove original LMP data since it is large
        os.remove(lmp_fname)

def download(year:int, node_id: str, auxiliary: bool, is_rt_data: bool):
    """ See script entry (at the bottom) for the meaning """

    if auxiliary:
        items = ['demand', 'renewable', 'actual_renewable', 'actual_demand']
        folder = "COMMON"
    else:
        items = ['lmp']
        folder = node_id

    if not os.path.exists(folder):
        os.mkdir(folder)

    days_in_month = [
        (12, 31, year-1),
        (1, 31, year),
        (2, 29 if year % 4 == 0 else 28, year),
        (3, 31, year),
        (4, 30, year),
        (5, 31, year),
        (6, 30, year),
        (7, 31, year),
        (8, 31, year),
        (9, 30, year),
        (10, 31, year),
        (11, 30, year),
    ]
        
    startdates = []
    enddates = []
    for (mo, num_days, yr) in days_in_month:
        if num_days >= 31:
            startdate = "%i%02d01" % (yr, mo)
            enddate = "%i%02d31" % (yr, mo)
            startdates.append(startdate)
            enddates.append(enddate)

            startdate = "%i%02d31" % (yr, mo)
            enddate = "%i%02d01" % (yr+1 if mo == 12 else yr, (mo % 12) + 1)
            startdates.append(startdate)
            enddates.append(enddate)

        else:
            startdate = "%i%02d01" % (yr, mo)
            enddate = "%i%02d01" % (yr, (mo % 12) + 1)
            startdates.append(startdate)
            enddates.append(enddate)

    for item in items:
        _download(node_id, item, folder, startdates, enddates, is_rt_data)
    if 'lmp' in items:
        _clean(node_id, startdates, enddates, is_rt_data)

if __name__ == "__main__":
    """ Works as of Sep 1st, 2024. 

    Helpful links:
      API: https://www.caiso.com/Documents/OASIS-InterfaceSpecification_v5_1_1Clean_Fall2017Release.pdf
      pnode map: https://www.caiso.com/todays-outlook/prices
      another pnode map: http://electricitymapper.appspot.com
      interzonal map: https://www.caiso.com/Documents/Chapter5_Inter-ZonalCongestionManagementMarket.pdf

    :args year: Calendar year to download data from
    :args node_id: pnode to download data from:
      ALAMT3G_7_B1, ALAMT3G_7_N002, COTWDPGE_1_N001, COTWDPGE_1_N024, 
      FREMNT_1_N013, MOSSLND7_7_B2, MOSSLND7_7_ND001, PAULSWT_1_N004, 
      PAULSWT_1_N013
    :args auxiliary: If true, downloads data other than RT LMPs. Otherwise downloads RT LMPs
    :pargs dam: If request LMP, downloads day-ahead market LMP. Otherwise, RT
    """
    parser = argparse.ArgumentParser(
        prog='Download CAISO',
        description='Downloads CAISO data',
    )

    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--node_id', type=str)
    parser.add_argument('--auxiliary', action="store_true")
    parser.add_argument('--dam_mkt', action="store_true")
    args = parser.parse_args()

    if args.node_id is not None:
        check_pnode_id(args.node_id)

    download(args.year, args.node_id, args.auxiliary, not args.dam_mkt)
