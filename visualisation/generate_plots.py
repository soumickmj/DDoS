#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Step 3 of 3

def convertInp2Out(df, method_name):
    df = df[df.columns.drop(list(df.filter(regex='Out')))]
    df.columns = df.columns.str.replace('Inp', 'Out')
    df.model = method_name
    return df

samplings = {
    "Center4MaskWoPad": "4% of kSpace",
    "Center6p25MaskWoPad": "6.25% of kSpace",
    "Center10MaskWoPad": "10% of kSpace",
}

models = {
    "ZeroPadded": "Zero-padded",
    "Baseline_Dyn": "UNet (CHAOS Dynamic)",
    "Baseline_NonDyn": "UNet (CHAOS)",
    "DDoS": "DDoS-UNet",
}

subjects = {
    "PhilAbd3DDyn1conST": 0,
    "MarioAbd3DDyn1conST": 1,
    "MickAbd3DDyn3conST": 2,
    "ChimpAbd3DDyn3conST": 3,
    "FatyAbd3DDyn3conST": 4,
}

legend_order = ["Interpolated Input", "Zero-padded", "UNet (CHAOS)", "UNet (CHAOS Dynamic)", "DDoS-UNet"]

consolidated_csv = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Results/DDoS_Paper1/dynDualChn/DDoS-UNet/FullVol/woZPad/Results/consolidated.csv"
results = pd.read_csv(consolidated_csv)
results = results.replace({"model": models, "undersampling": samplings, "subject":subjects})
results = results.rename(columns={'subject': 'SubjectID', 'undersampling': 'Undersampling'})

def generate_plot(df, y, path, title="", x_marker="Timepoint", ci=95):
    ax = sns.boxplot(data=df, x=x_marker, y=y, hue="Method", palette=("pastel"))
    # if x_marker == "Timepoint":
    #     ax = sns.lineplot(data=df, x=x_marker, y=y, hue="Method", style="SubjectID", marker="o")
    # else:
    #     ax = sns.lineplot(data=df, x=x_marker, y=y, hue="Method", marker="o")
    # ax = sns.lineplot(data=df, x=x_marker, y=y, hue="Method", marker="o", ci=ci)
    # ax = sns.relplot(data=df, x=x_marker, y=y, hue="Method")
    # ax = sns.scatterplot(data=df, x=x_marker, y=y, hue="Method")
    
    if x_marker == "SubjectID":
        #fix the x-ticks
        ax.set(xticks=df[x_marker].unique())

    #re-arrange legends
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in legend_order]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.01, 1),borderaxespad=0) 

    #plot and save
    plt.title(title,fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, format='png')
    # plt.show()
    plt.clf()

for us in samplings.keys():
    us_results = results[results["Undersampling"] == samplings[us]].sort_values(by ='file')

    m0 = us_results.model.unique()[0]
    trilinear_df = convertInp2Out(us_results[us_results.model == m0], "Interpolated Input")

    df = pd.concat([us_results, trilinear_df])
    df = df.reset_index(drop=True)
    df = df.rename(columns={'SSIMOut': 'SSIM', 'PSNROut': 'PSNR', "model": "Method", "file":"Timepoint"})

    df.sort_values('Method',inplace=True, ascending=False)

    TPtags = df.Timepoint.str.split("_").str[0].unique()
    for tp in TPtags:
        df = df.replace(to_replace =tp+'_*', value = int(tp.replace("TP","")), regex = True)

    generate_plot(df, y='SSIM', x_marker="SubjectID", path=consolidated_csv.replace("consolidated.csv","Plots")+'/xSubjectBox_'+us+'_SSIM.png', title=samplings[us])
    generate_plot(df, y='PSNR', x_marker="SubjectID", path=consolidated_csv.replace("consolidated.csv","Plots")+'/xSubjectBox_'+us+'_PSNR.png', title=samplings[us])

    #generate_plot(df, y='SSIM', x_marker="Timepoint", path=consolidated_csv.replace("consolidated.csv","Plots")+'/xTP_'+us+'_SSIM.png', title=samplings[us], ci=None)
    #generate_plot(df, y='PSNR', x_marker="Timepoint", path=consolidated_csv.replace("consolidated.csv","Plots")+'/xTP_'+us+'_PSNR.png', title=samplings[us], ci=None)