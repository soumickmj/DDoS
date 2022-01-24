#!/bin/bash

#Params: 
#1: GPUID
#2: Subject Name
#3: Protocol ID

datasetpath="/home/schatter/Soumick/Data/Chimp/3DDynTest/$2Abdomen3DDyn/DynProtocol$3/Filtered/"
ddosouttype="StatTPinitCumulative_$2Abd3DDyn$3conST_woZpad_full_Best"
baselineouttype="StatTPinitIgnored_$2Abd3DDyn$3conST_woZpad_full_Best"
outpathnondyn="/home/schatter/Soumick/Data/Chimp/CHAOSwoT2/"

echo "Analysing Dataset...."
CUDA_VISIBLE_DEVICES=$1 python analyse_dataset.py --dataset $datasetpath --outtype $ddosouttype --us "Center4Mask" 
CUDA_VISIBLE_DEVICES=$1 python analyse_dataset.py --dataset $datasetpath --outtype $ddosouttype --us "Center6p25Mask" 
CUDA_VISIBLE_DEVICES=$1 python analyse_dataset.py --dataset $datasetpath --outtype $ddosouttype --us "Center10Mask"
echo "Analysing Dataset...."