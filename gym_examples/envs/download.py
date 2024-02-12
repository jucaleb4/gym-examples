import pandas as pd
import requests, zipfile, io
import time

def download(node_id: str):

    startdates = ["601", "701", "731"]
    enddates = ["701", "731", "830"]

    for (startdate, enddate) in zip(startdates, enddates):

        startdatetime=f"20230{startdate}T08:00-0000"
        enddatetime=f"20230{enddate}T08:00-0000"

        # for RT LMPs
        # url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_RTPD_LMP&startdatetime={startdatetime}&enddatetime={enddatetime}&version=2&resultformat=6&market_run_id=RTPD&node={node_id}"

        # demand 
        # url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=DAM"

        # solar and wind
        url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_REN_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=DAM"
        # url = f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_REN_FCST&startdatetime={startdatetime}&enddatetime={enddatetime}&version=1&resultformat=6&market_run_id=RTPD"

        import ipdb; ipdb.set_trace()
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(".")

        time.sleep(1)

# TAC and REN: TAC_ECNTR, SP15
node_id = "MIL1_3_PASGNODE" # for price 
# download(node_id)

fname = "20230601_20230701_SLD_FCST_DAM_20230927_12_29_00_v1.csv"

# print(f"Reading file {fname}")
df = pd.read_csv(fname)

import ipdb; ipdb.set_trace()
print(df.head())
# print(df.describe())
# print(df.info())
