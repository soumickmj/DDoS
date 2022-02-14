import pandas as pd

total = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Results/DDoS_Paper1/dynDualChn/DDoS-UNet/FullVol/Results/consolidated_wrong_diffSDZeroPad.csv"
zero = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Results/DDoS_Paper1/dynDualChn/DDoS-UNet/FullVol/Results/consolidated_zeroP.csv"
newcsv = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Results/DDoS_Paper1/dynDualChn/DDoS-UNet/FullVol/Results/consolidated.csv"

df = pd.read_csv(total)
print(len(df))
df = df[df.model != "ZeroPadded"]
print(len(df))

zero_df = pd.read_csv(zero)
df = pd.concat([df, zero_df])
print(len(df))

df.to_csv(newcsv)