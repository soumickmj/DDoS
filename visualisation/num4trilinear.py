from glob import glob
import torch
from tqdm import tqdm
import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch.nn.functional as F

from utils.utilities import calc_metircs

fully_root = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Data/3DDynTest/MickAbdomen3DDyn/DynProtocol3/Filtered/hrTestDynConST"
under_root = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Data/3DDynTest/MickAbdomen3DDyn/DynProtocol3/Filtered/usTestDynConST"
interp = "trilinear"

files = sorted(glob(f"{under_root}/**/*.nii.gz", recursive=True))

metrics = []
for f in tqdm(files):
    if ("Center" not in f and "Centre" not in f) or "TP00" in f or "WoPad" not in f:
        continue
    fully_parts = f.replace(under_root, fully_root).split(os.path.sep)
    undersampling = fully_parts[-3]
    del fully_parts[-3]
    f_fully = os.path.sep.join(fully_parts)

    vol_under = np.array(nib.load(f).get_fdata())
    vol_fully = np.array(nib.load(f_fully).get_fdata())

    vol_under /= vol_under.max()
    vol_fully /= vol_fully.max()

    vol_under = F.interpolate(torch.from_numpy(vol_under).unsqueeze(0).unsqueeze(0), size=vol_fully.shape, mode=interp, align_corners=False).squeeze().numpy()

    inp_metrics, inp_ssimMAP = calc_metircs(vol_fully, vol_under, tag="ZPad")

    inp_metrics["file"] = fully_parts[-2] + "_" + fully_parts[-1]
    inp_metrics["subject"] = fully_parts[-6] + "_" + fully_parts[-5]
    inp_metrics["undersampling"] = undersampling + "WoPad"
    inp_metrics["model"] = interp.capitalize()
    inp_metrics["DiffZPad"] = np.std(vol_fully - vol_under)

    metrics.append(inp_metrics)
df = pd.DataFrame.from_dict(metrics)
df.to_csv(f"{os.path.dirname(under_root)}/noprevnorm_metrics_{interp}.csv")