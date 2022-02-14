#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

sns.set_theme(style="darkgrid")

#Step 4 (actual) of 4

def convertInp2Out(df, method_name):
    df = df[df.columns.drop(list(df.filter(regex='Out')))]
    df.columns = df.columns.str.replace('Inp', 'Out')
    df.model = method_name
    return df

samplings = {
    "Center4MaskWoPad": "4% of k-space",
    "Center6p25MaskWoPad": "6.25% of k-space",
    "Center10MaskWoPad": "10% of k-space",
}

models = {
    "Trilinear": "Trilinear Interpolation",
    "ZeroPadded": "Zero-padded",
    "Baseline_Dyn": "UNet\n(CHAOS\nDynamic)",
    "Baseline_NonDyn": "UNet\n(CHAOS)",
    "DDoS": "DDoS-UNet",
}

subjects = {
    "PhilAbd3DDyn1conST": 0,
    "MarioAbd3DDyn1conST": 1,
    "MickAbd3DDyn3conST": 2,
    "ChimpAbd3DDyn3conST": 3,
    "FatyAbd3DDyn3conST": 4,
}

<<<<<<< Updated upstream
legend_order = ["Interpolated\nInput", "Zero-padded", "UNet\n(CHAOS)", "UNet\n(CHAOS\nDynamic)", "DDoS-UNet"]
=======
legend_order = ["Trilinear Interpolation", "Zero-padded", "UNet (CHAOS)", "UNet (CHAOS Dynamic)", "DDoS-UNet"]
>>>>>>> Stashed changes

ignore_antipasto = False
consolidated_csv = "/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Chompunuch/PhD/Results/DDoS_Paper1/dynDualChn/DDoS-UNet/FullVol/woZPad/Results/QuantitativeAnalysis/Sources/consolidated.csv"
results = pd.read_csv(consolidated_csv)
results = results.replace({"model": models, "undersampling": samplings, "subject":subjects})
results = results.rename(columns={'subject': 'SubjectID', 'undersampling': 'Undersampling'})

<<<<<<< Updated upstream
def generate_plot(df, y, path, title="", x_marker="Timepoint", hue="Method", ci=95, plot_type="line", grid_style="darkgrid"):
    if plot_type == "box":
        sns.set_theme(style=grid_style)
        ax = sns.boxplot(data=df, x=x_marker, y=y, hue=hue, palette=("pastel"), order=legend_order)
        # plt.xticks(rotation = 90)
    else:
        sns.set_theme(style=grid_style)
        df.Method = df.Method.str.replace("\n", " ")
        ax = sns.lineplot(data=df, x=x_marker, y=y, hue=hue, marker="o", ci=ci)
=======
def generate_boxplot(df, y, path, title="", x_marker="Method", palette=("pastel")):
    df.Method = df.Method.str.replace(" ", "\n")
    ax = sns.boxplot(data=df, x=x_marker, y=y, palette=palette, order=[l.replace(" ", "\n") for l in legend_order]) 

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    #plot and save
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(title,fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, format='png')
    # plt.show()
    plt.clf()

def generate_lineplot(df, y, path, title="", x_marker="Timepoint", ci=95):
    # if x_marker == "Timepoint":
    #     ax = sns.lineplot(data=df, x=x_marker, y=y, hue="Method", style="SubjectID", marker="o")
    # else:
    #     ax = sns.lineplot(data=df, x=x_marker, y=y, hue="Method", marker="o")
    ax = sns.lineplot(data=df, x=x_marker, y=y, hue="Method", marker="o", ci=ci)
    # ax = sns.relplot(data=df, x=x_marker, y=y, hue="Method")
    # ax = sns.scatterplot(data=df, x=x_marker, y=y, hue="Method")
>>>>>>> Stashed changes
    
    if x_marker == "SubjectID":
        #fix the x-ticks
        ax.set(xticks=df[x_marker].unique())

    #re-arrange legends
    if bool(hue):
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(l.replace("\n", " ")) for l in legend_order]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.01, 1),borderaxespad=0) 

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    #plot and save
    plt.xticks(fontsize=10)
<<<<<<< Updated upstream
=======
    plt.yticks(fontsize=10)
>>>>>>> Stashed changes
    plt.title(title,fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(path, format='png')
    plt.clf()

for us in samplings.keys():
    us_results = results[results["Undersampling"] == samplings[us]].sort_values(by ='file')

<<<<<<< Updated upstream
    m0 = us_results.model.unique()[0]
    trilinear_df = convertInp2Out(us_results[us_results.model == m0], "Interpolated\nInput")
=======
    # m0 = us_results.model.unique()[0]
    # trilinear_df = convertInp2Out(us_results[us_results.model == m0], "Trilinear Interpolation")
>>>>>>> Stashed changes

    # df = pd.concat([us_results, trilinear_df])
    df = us_results
    df = df.reset_index(drop=True)
    df = df.rename(columns={'SSIMOut': 'SSIM', 'PSNROut': 'PSNR', "model": "Method", "file":"Timepoint"})

    df.sort_values('Method',inplace=True, ascending=False)

    TPtags = df.Timepoint.str.split("_").str[0].unique()
    for tp in TPtags:
        df = df.replace(to_replace =tp+'_*', value = int(tp.replace("TP","")), regex = True)

<<<<<<< Updated upstream
    # generate_plot(df, y='SSIM', x_marker="SubjectID", path=consolidated_csv.replace("consolidated.csv","Plots")+'/xSubject_'+us+'_SSIM.png', title=samplings[us])
    # generate_plot(df, y='PSNR', x_marker="SubjectID", path=consolidated_csv.replace("consolidated.csv","Plots")+'/xSubject_'+us+'_PSNR.png', title=samplings[us])

    generate_plot(df, y='SSIM', x_marker="Timepoint", path=consolidated_csv.replace("consolidated.csv","Plots")+'/xTPLine_'+us+'_SSIM.png', title=samplings[us], ci=None)
    generate_plot(df, y='PSNR', x_marker="Timepoint", path=consolidated_csv.replace("consolidated.csv","Plots")+'/xTPLine_'+us+'_PSNR.png', title=samplings[us], ci=None)

    # generate_plot(df, y='SSIM', x_marker="Method", hue=None, path=consolidated_csv.replace("consolidated.csv","Plots")+'/xSubjectBoxDark_'+us+'_SSIM.png', title=samplings[us], plot_type="box")
    # generate_plot(df, y='PSNR', x_marker="Method", hue=None, path=consolidated_csv.replace("consolidated.csv","Plots")+'/xSubjectBoxDark_'+us+'_PSNR.png', title=samplings[us], plot_type="box")

    # generate_plot(df, y='SSIM', x_marker="Method", hue=None, path=consolidated_csv.replace("consolidated.csv","Plots")+'/xSubjectBox_'+us+'_SSIM.png', title=samplings[us], plot_type="box", grid_style="whitegrid")
    # generate_plot(df, y='PSNR', x_marker="Method", hue=None, path=consolidated_csv.replace("consolidated.csv","Plots")+'/xSubjectBox_'+us+'_PSNR.png', title=samplings[us], plot_type="box", grid_style="whitegrid")
=======
    if ignore_antipasto:
        df = df[df.Timepoint!=df.Timepoint.unique().min()]

    # generate_plot(df, y='SSIM', x_marker="SubjectID", path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/xSubjectBox_'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_SSIM.png', title=samplings[us])
    # generate_plot(df, y='PSNR', x_marker="SubjectID", path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/xSubjectBox_'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_PSNR.png', title=samplings[us])

    
    # #DO THIS
    # generate_lineplot(df, y='SSIM', x_marker="Timepoint", path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/xTP_'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_SSIM.png', title=samplings[us], ci=None)
    # generate_lineplot(df, y='PSNR', x_marker="Timepoint", path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/xTP_'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_PSNR.png', title=samplings[us], ci=None)
    
    
    #generate_plot(df, y='PSNR', x_marker="Timepoint", path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/xTP_'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_PSNR.png', title=samplings[us], ci=None)

    # generate_plot(df, y='SSIM', x_marker="Timepoint", path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/ciTP_'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_SSIM.png', title=samplings[us], ci=95)
    # generate_plot(df, y='PSNR', x_marker="Timepoint", path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/ciTP_'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_PSNR.png', title=samplings[us], ci=95)

    # #DO THIS
    # generate_lineplot(df, y='SSIM', x_marker="SubjectID", path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/ciSubwise'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_SSIM.png', title=samplings[us], ci=95)
    # generate_lineplot(df, y='PSNR', x_marker="SubjectID", path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/ciSubwise'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_PSNR.png', title=samplings[us], ci=95)
    
    
    # generate_plot(df, y='SSIM', x_marker="SubjectID", path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/xSubwise_'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_SSIM.png', title=samplings[us], ci=None)
    # generate_plot(df, y='PSNR', x_marker="Timepoint", path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/ciTP_'+us+'_PSNR.png', title=samplings[us], ci=95)

    # #DO THIS
    # generate_boxplot(df, y='SSIM', path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/xMethodBox_'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_SSIM.png', title=samplings[us])
    generate_boxplot(df, y='PSNR', path=consolidated_csv.replace("Sources/consolidated.csv", "Plots")+'/xMethodBox_'+('_noAP_' if ignore_antipasto else '')+'_'+us+'_PSNR.png', title=samplings[us])

    
>>>>>>> Stashed changes
