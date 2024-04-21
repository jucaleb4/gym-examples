import argparse
import pandas as pd
import requests, zipfile, io
import time
import os

def _download(node_id: str, item: str, folder:str):
    """ 
    Downloads data into designated folder

    :param node_id: which node in CAISO to retrieve LMPs
    :param item: which query item (lmp, demand, renewable)
    :param folder: which folder to extract the files into
    """
    assert os.path.exists(folder), "Must have folder %s before downloading" % folder

    startdates = ["601", "701", "731", "801", "831"]
    enddates = ["701", "731", "801", "831", "901"]

    for (startdate, enddate) in zip(startdates, enddates):

        startdatetime=f"20230{startdate}T08:00-0000"
        enddatetime=f"20230{enddate}T08:00-0000"

        # for RT LMPs
        if item == "lmp":
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
        sleep_time = max(10 - e_time, 0) + 1
        time.sleep(sleep_time)

def download(node_id: str, auxilary: bool):
    """
    Creates download folder and downloads item. See `_download` for more info.

    Example node_id: 'MIL1_3_PASGNODE'

    TODO: How to identify TAC and region?
    TAC and REN: TAC_ECNTR, SP15
    """

    if auxilary:
        items = ['demand', 'renewable', 'actual_renewable']
        folder = "COMMON"
    else:
        items = ['lmp']
        if node_id is None:
            print("Please pass a node_id if you do not want auxiliary data")
            return
        folder = node_id

    # make the folder
    if not os.path.exists(folder):
        os.mkdir(folder)

    for item in items:
        _download(node_id, item, folder)

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

    parser.add_argument('--node_id', type=str)
    parser.add_argument('--auxilary', action="store_true")
    args = parser.parse_args()

    download(args.node_id, args.auxilary)
