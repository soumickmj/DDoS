from glob import glob
from tqdm import tqdm
import os
import nibabel as nib
import numpy as np
import pandas as pd

from utils.utilities import calc_metircs

fully_root = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Data/3DDynTest/MarioAbdomen3DDyn/DynProtocol1/Filtered/hrTestDynConST"
zpad_root = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Data/3DDynTest/MarioAbdomen3DDyn/DynProtocol1/Filtered/usTestDynConST"

files = sorted(glob(f"{zpad_root}/**/*.nii.gz", recursive=True))

metrics = []
for f in tqdm(files):
    if ("Center" not in f and "Centre" not in f) or "TP00" in f or "WoPad" in f:
        continue
    fully_parts = f.replace(zpad_root, fully_root).split(os.path.sep)
    undersampling = fully_parts[-3]
    del fully_parts[-3]
    f_fully = os.path.sep.join(fully_parts)

    vol_zpad = np.array(nib.load(f).get_fdata())
    vol_fully = np.array(nib.load(f_fully).get_fdata())

    vol_zpad /= vol_zpad.max()
    vol_fully /= vol_fully.max()

    inp_metrics, inp_ssimMAP = calc_metircs(vol_fully, vol_zpad, tag="ZPad")

    inp_metrics["file"] = fully_parts[-2] + "_" + fully_parts[-1]
    inp_metrics["subject"] = fully_parts[-6] + "_" + fully_parts[-5]
    inp_metrics["undersampling"] = undersampling + "WoPad"
    inp_metrics["model"] = "ZeroPadded"
    inp_metrics["DiffZPad"] = np.std(vol_fully - vol_zpad)

    metrics.append(inp_metrics)
df = pd.DataFrame.from_dict(metrics)
df.to_csv(os.path.dirname(zpad_root)+"/noprevnorm_metrics_zpad.csv")