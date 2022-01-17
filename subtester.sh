#!/bin/bash

#Params: 
#1: GPUID
#2: Subject Name
#3: Protocol ID

datasetpath="/home/schatter/Soumick/Data/Chimp/3DDynTest/$2Abdomen3DDyn/DynProtocol$3/Filtered/"
ddosouttype="StatTPinitCumulative_$2Abd3DDyn$3conST_woZpad_full_Best"
baselineouttype="StatTPinitIgnored_$2Abd3DDyn$3conST_woZpad_full_Best"
outpathnondyn="/home/schatter/Soumick/Data/Chimp/CHAOSwoT2/"

echo "Testing DDoS...."
CUDA_VISIBLE_DEVICES=$1 python apply_DDoS_full.py --dataset $datasetpath --outtype $ddosouttype --us "Center4MaskWoPad" --modelname "usTrain_UNETfulldo0.0dp3upsample_Center4MaskWoPad_pLossL1lvl3"
CUDA_VISIBLE_DEVICES=$1 python apply_DDoS_full.py --dataset $datasetpath --outtype $ddosouttype --us "Center6p25MaskWoPad" --modelname "usTrain_UNETfulldo0.0dp3upsample_Center6p25MaskWoPad_pLossL1lvl3"
CUDA_VISIBLE_DEVICES=$1 python apply_DDoS_full.py --dataset $datasetpath --outtype $ddosouttype --us "Center10MaskWoPad" --modelname "usTrain_UNETfulldo0.0dp3upsample_Center10MaskWoPad_pLossL1lvl3"
echo "Testing DDoS Finished...."

echo "Testing dynamic baseline...."
CUDA_VISIBLE_DEVICES=$1 python apply_DDoS_full_baseline.py --dataset $datasetpath --outtype $baselineouttype --us "Center4MaskWoPad" --modelname "usTrain_UNETfullBaselinedo0.0dp3upsample_Center4MaskWoPad_pLossL1lvl3"
CUDA_VISIBLE_DEVICES=$1 python apply_DDoS_full_baseline.py --dataset $datasetpath --outtype $baselineouttype --us "Center6p25MaskWoPad" --modelname "usTrain_UNETfullBaselinedo0.0dp3upsample_Center6p25MaskWoPad_pLossL1lvl3"
CUDA_VISIBLE_DEVICES=$1 python apply_DDoS_full_baseline.py --dataset $datasetpath --outtype $baselineouttype --us "Center10MaskWoPad" --modelname "usTrain_UNETfullBaselinedo0.0dp3upsample_Center10MaskWoPad_pLossL1lvl3"
echo "Testing dynamic baseline Finished...."

echo "Testing non-dynamic baseline...."
CUDA_VISIBLE_DEVICES=$1 python apply_DDoS_full_baseline.py --dataset $datasetpath --outtype $baselineouttype --us "Center4MaskWoPad" --outpath $outpathnondyn --modelname "usTrain_UNETfullBaselineNonDyndo0.0dp3upsample_Center4MaskWoPad_pLossL1lvl3"
CUDA_VISIBLE_DEVICES=$1 python apply_DDoS_full_baseline.py --dataset $datasetpath --outtype $baselineouttype --us "Center6p25MaskWoPad" --outpath $outpathnondyn --modelname "usTrain_UNETfullBaselineNonDyndo0.0dp3upsample_Center6p25MaskWoPad_pLossL1lvl3"
CUDA_VISIBLE_DEVICES=$1 python apply_DDoS_full_baseline.py --dataset $datasetpath --outtype $baselineouttype --us "Center10MaskWoPad" --outpath $outpathnondyn --modelname "usTrain_UNETfullBaselineNonDyndo0.0dp3upsample_Center10MaskWoPad_pLossL1lvl3"
echo "Testing non-dynamic baseline Finished...."
