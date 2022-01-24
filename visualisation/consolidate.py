import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import nibabel as nib

def MinMax(data):
    return (data-data.min())/(data.max()-data.min())

#Step 1 (actual) of 3

results_root = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Results/DDoS_Paper1/dynDualChn/DDoS-UNet/FullVol/ZPad/Results"
dataset_root = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Data/3DDynTest"

results_csvs = glob(f"{results_root}/**/Results.csv", recursive=True)

dfs = []
for csv in tqdm(results_csvs):
    # if "ZeroPadded" not in csv:
    #     continue
    df = pd.read_csv(csv)
    fileparts = csv.split(os.path.sep)
    if "ZeroPadded" not in csv:
        recontype, subject, _, _, _ = fileparts[-2].split("_")
        undersampling = fileparts[-3].split("_")[2]
        model = fileparts[-4]
    else:
        recontype, subject, _, _, _, undersampling = fileparts[-2].split("_")
        model = "ZeroPadded"
        undersampling += "WoPad"
        df.columns = df.columns.str.replace('Inp', 'Out')
    df["recontype"] = recontype
    df["subject"] = subject
    df["undersampling"] = undersampling
    df["model"] = model
    df['DiffInp'] = 1.0
    df['DiffOut'] = 1.0

    datasetpath=f"{dataset_root}/{subject.split('Abd3DDyn')[0]}Abdomen3DDyn/DynProtocol{subject.split('Abd3DDyn')[1][0]}/Filtered/hrTestDyn{subject.split('Abd3DDyn')[1][1:].replace('conST', 'ConST')}"
    for f in df.file.unique():
        tp = f.split("_")[0]

        tppath = glob(f"{datasetpath}/{tp}/*.nii*")[0]
        gt = MinMax(nib.load(tppath).get_fdata())

        try:
            inppath = glob(csv.replace(".csv",f"/{f}/inp.nii*"))[0]
            inp = nib.load(inppath).get_fdata()
            diff_inp = gt - inp
            nib.save(nib.Nifti1Image(diff_inp, np.eye(4)), csv.replace(".csv",f"/{f}/diff_inp_nonorm.nii.gz"))
            diff_inp_std = np.std(diff_inp)
            if "ZeroPadded" not in csv:
                df.loc[df.file==f, "DiffInp"] = diff_inp_std
            else:
                df.loc[df.file==f, "DiffOut"] = diff_inp_std
        except:
            pass
        
        try:
            outpath = glob(csv.replace(".csv",f"/{f}/out.nii*"))[0]
            out = nib.load(outpath).get_fdata()
            diff_out = gt - out
            nib.save(nib.Nifti1Image(diff_out, np.eye(4)), csv.replace(".csv",f"/{f}/diff_out_nonorm.nii.gz"))
            diff_out_std = np.std(diff_out)
            df.loc[df.file==f, "DiffOut"] = diff_out_std
        except:
            pass
    dfs.append(df)


df = pd.concat(dfs)
df.to_csv(f"{results_root}/consolidated.csv")