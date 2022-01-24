import pandas as pd
from glob import glob
from tqdm import tqdm
import os
from scipy.stats import mannwhitneyu

#Step 2 of 3

# consolidated_csv = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Results/DDoS_Paper1/dynDualChn/DDoS-UNet/FullVol/Results/consolidated.csv"
consolidated_csv = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Results/DDoS_Paper1/dynDualChn/DDoS-UNet/FullVol/ZPad/Results/consolidated_baseNonDyn.csv"
df = pd.read_csv(consolidated_csv)
target_df = df[df.model=="DDoS"]

def getStrings(mean, median, std, metric):    
    avg = str(mean[metric].round(3).apply(str).str.cat(std[metric].round(3).apply(str), sep="±"))
    med = str(median[metric])
    return avg, med

def getInfo(df, groupby, metric_type="Out"):
    mean = df.groupby(groupby).mean()
    median = df.groupby(groupby).median()
    std = df.groupby(groupby).std()
    avgstrSSIM, medstrSSIM = getStrings(mean, median, std, 'SSIM'+metric_type)
    avgstrNRMSE, medstrNRMSE = getStrings(mean, median, std, 'NRMSE'+metric_type)
    avgstrPSNR, medstrPSNR = getStrings(mean, median, std, 'PSNR'+metric_type)
    avgstrDiff, medstrDiff = getStrings(mean, median, std, 'Diff'+metric_type)
    return (avgstrSSIM, medstrSSIM), (avgstrNRMSE, medstrNRMSE), (avgstrPSNR, medstrPSNR), (avgstrDiff, medstrDiff)

def writeIndividual(file_obj, val, metric):
    file_obj.write("\n----------------------------\n")
    file_obj.write("\n")
    file_obj.write(f"\nAverage {metric}:\n")
    file_obj.write(val[0])
    file_obj.write(f"\nMedian {metric}:\n")
    file_obj.write(val[1])
    file_obj.write("\n----------------------------\n")
    file_obj.write("\n")

def writeSubDF(df, target_df, file_obj, model, groupby, metric_type="Out"):
    SSIM, NRMSE, PSNR, Diff = getInfo(df, groupby, metric_type)
    
    file_obj.write("\n----------------------------\n")
    file_obj.write("\n----------------------------\n")
    file_obj.write(f"\nModel: {model}\n")
    file_obj.write("\n----------------------------\n")
    file_obj.write("\n----------------------------\n")
    file_obj.write("\n")
    writeIndividual(file_obj, SSIM, "SSIM")
    file_obj.write(f"\np-value: {getP(df, target_df, metric='SSIM', metric_type=metric_type)}\n")
    writeIndividual(file_obj, NRMSE, "NRMSE")
    file_obj.write(f"\np-value: {getP(df, target_df, metric='NRMSE', metric_type=metric_type)}\n")
    writeIndividual(file_obj, PSNR, "PSNR")
    file_obj.write(f"\np-value: {getP(df, target_df, metric='PSNR', metric_type=metric_type)}\n")
    writeIndividual(file_obj, Diff, "Diff")
    file_obj.write(f"\np-value: {getP(df, target_df, metric='Diff', metric_type=metric_type)}\n")
    file_obj.write("\n----------------------------\n")
    file_obj.write("\n----------------------------\n")

def getP(model_df, target_df, metric, metric_type="Out"):
    pstring = "\n"
    if len(target_df) > 0:
        for us in model_df.undersampling.unique():    
            target = target_df[target_df.undersampling==us][metric+metric_type]    
            current = model_df[model_df.undersampling==us][metric+"Out"]    
            pstring += us + ": " + str(mannwhitneyu(target,current).pvalue) + "\n"
        return pstring
    else:
        return "-1"

#Model-wise undersampling scores
with open(f"{os.path.dirname(consolidated_csv)}/scores_model_undersampling.txt","w") as file_obj:
    m0 = df.model.unique()[0]
    model_df = df[df.model == m0]
    writeSubDF(model_df, target_df, file_obj, "Trilinear", "undersampling", metric_type="Inp")

    for m in df.model.unique():
        model_df = df[df.model == m]
        writeSubDF(model_df, target_df, file_obj, m, "undersampling", metric_type="Out")

#Subject Model-wise undersampling scores
with open(f"{os.path.dirname(consolidated_csv)}/scores_subject_model_undersampling.txt","w") as file_obj:
    for s in df.subject.unique():
        sub_df = df[df.subject == s]
        sub_target_df = target_df[target_df.subject == s]
        file_obj.write("\n§§§§§§§§§§§§§§§§§§§§§§§§§§§§\n")
        file_obj.write("\n§§§§§§§§§§§§§§§§§§§§§§§§§§§§\n")
        file_obj.write(f"\nSubject: {s}\n")
        file_obj.write("\n§§§§§§§§§§§§§§§§§§§§§§§§§§§§\n")
        file_obj.write("\n§§§§§§§§§§§§§§§§§§§§§§§§§§§§\n")
        file_obj.write("\n")

        m0 = sub_df.model.unique()[0]
        model_df = sub_df[sub_df.model == m0]
        writeSubDF(model_df, sub_target_df, file_obj, "Trilinear", "undersampling", metric_type="Inp")

        for m in sub_df.model.unique():
            model_df = sub_df[sub_df.model == m]
            writeSubDF(model_df, sub_target_df, file_obj, m, "undersampling", metric_type="Out")
        
        