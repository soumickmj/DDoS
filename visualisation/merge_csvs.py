import pandas as pd
from glob import glob
import os

#Step 2 (actual) of 4

csv_root = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Results/DDoS_Paper1/dynDualChn/DDoS-UNet/FullVol/woZPad/Results/QuantitativeAnalysis/CSVs"

dfDL = pd.read_csv(f"{csv_root}/RAWs/DL_Results.csv")
dfDL.drop(dfDL[dfDL.model == "ZeroPadded"].index, inplace=True)
dfDL.reset_index(drop=True, inplace=True)

for f in glob(f"{csv_root}/RAWs/*.csv"):
    if "DL_Results" in f:
        continue
    subID = os.path.basename(f).split("_")[0]
    if "zpad" in f:
        model = "ZeroPadded"
    else:
        model = os.path.basename(f).split("_")[-1].split(".")[0].capitalize()
    df = pd.read_csv(f)
    df.subject = subID
    df.model = model

    newcols = {
        c: c.split("ZPad")[0] + "Out"
        for c in [c for c in df.columns if "ZPad" in c]
    }

    df.rename(columns=newcols, inplace=True)
    df.undersampling = df.undersampling.str.replace("WoPadWoPad", "WoPad")
    df.drop(df[df.undersampling == "Center4Mask2WoPad"].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    dfDL = dfDL.append(df)
    dfDL.reset_index(drop=True, inplace=True)
dfDL.to_csv(f"{csv_root}/consolidated.csv")
