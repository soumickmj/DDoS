import argparse
import logging
import math
import os
import random
import statistics
import sys

import numpy as np
import pandas as pd
import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb
from models import *
from models.ReconResNet import ResNet
from models.ShuffleUNet.net import ShuffleUNet
from models.ThisNewNet import ThisNewNet
from utils.data import *
from utils.datasets_dyn import SRDataset
from utils.utilities import ResSaver

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2022, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee", "Chompunuch Sarasaen"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

modelIDs = {
    0: "UNET",
    1: "SRCNN",
    2: "SRCNNv2",
    3: "SRCNNv3",
    4: "UNETvSeg",
    5: "UNETvSegDS",
    6: "DenseNet",
    7: "UNETSRCNN",
    8: "SRCNNUNET",
    9: "ReconResNet",
    10: "ShuffleUNet",
    11: "UNETMSS",
}

lossIDs = {
    0: "pLoss",
    1: "MAE",
    2: "MultiSSIM",
    3: "SSIM3D"
}

def parseARGS():
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gpu", default="0", help="GPU ID(s).")
    ap.add_argument("--seed", default=2020, type=int, help="Seed") 
    ap.add_argument("-ds", "--dataset", default=r'/home/schatter/Soumick/Data/Chimp/3DDynTest/MickAbdomen3DDyn/DynProtocol3/Filtered/', help="Path to Dataset Folder.")
    ap.add_argument("-op", "--outpath", default=r'/home/schatter/Soumick/Data/Chimp/CHAOSwoT2Dyn/newSet/', help="Path for Output.")
    ap.add_argument("-ot", "--outtype", default=r'StatTPinit_MickAbd3DDyn3conST_woZpad_full_Best', help="Type of Recon currently being performed.")
    
    ap.add_argument("-us", "--us", default='Center4MaskWoPad', help="Undersample.")
    ap.add_argument("-s", "--scalefact", default='(1,1,1)', help="Scaling Factor. For Zero padded data, set the dim to 1. [As a 3 valued tuple, factor for each dim. Supply seperated by coma or as a tuple, no spaces in between.].")
    ap.add_argument("-uf", "--usfolder", default='usTestDynConST', help="Undersampled Folder.")
    ap.add_argument("-hf", "--hrfolder", default='hrTestDynConST', help="HighRes (Fully-sampled) Folder.") #hrTestDynPadded for ktGRASP
    ap.add_argument("-o", "--outfolder", default='dynDualChn', help="Output Folder.")

    ap.add_argument("-bs", "--batchsize", type=int, default=1, help="Batch Size.")
    ap.add_argument("-nw", "--nworkers", type=int, default=0, help="Number of Workers.")

    ap.add_argument("-m", "--modelname", default="ZeroPadded", help="Model to Load for testing.")
    # ap.add_argument("-bst", "--beststring", default="best", help="Model to Load for testing.")
    # ap.add_argument("-mb", "--modelbest", type=int, default=1, help="Model to Load for testing.")
    ap.add_argument("-c", "--cuda", type=bool, default=True, help="Use CUDA.")
    # ap.add_argument("-mg", "--mulgpu", type=bool, default=False, help="Use Multiple GPU.")
    ap.add_argument("-amp", "--amp", type=bool, default=True, help="Use AMP.")
    # ap.add_argument("-p", "--profile", type=bool, default=False, help="Do Model Profiling.")

    ap.add_argument("-ps", "--patchsize", default=None, help="Patch Size. Supply seperated by coma or as a tuple, no spaces in between. Set it to None if not desired.")
    ap.add_argument("-pst", "--patchstride", default='(3,3,3)', help="Stride of patches, to be used during validation")
    # ap.add_argument("-l", "--logfreq", type=int, default=10, help="log Frequency.")
    ap.add_argument("-ml", "--medianloss", type=int, default=True, help="Use Median to get loss value (Final Reduction).")

    # ap.add_argument("-inc", "--inchannel", type=int, default=1, help="Number of Channels in the Data.")
    # ap.add_argument("-otc", "--outchannel", type=int, default=1, help="Number of Channels in the Data.")
    ap.add_argument("-is", "--inshape", default='(256,256,30)', help="Input Shape. Supply seperated by coma or as a tuple, no spaces in between. Will only be used if Patch Size is None.")
    ap.add_argument("-int", "--preint", default="trilinear", help="Pre-interpolate before sending it to the Network. Set it to None if not needed.")    
    ap.add_argument("-nrm", "--prenorm", default=True, type=bool, help="Pre-norm before saving the images and calculating the metrics.")    

    ap.add_argument("-dus", "--detectus", type=int, default=0, help="Whether to replace the us using model name")

    #param to reproduce model
    # ap.add_argument("-mid", "--modelid", type=int, default=0, help="Model ID."+str(modelIDs))
    # ap.add_argument("-mbn", "--batchnorm", type=bool, default=False, help="(Only for Model ID 0, 11) Do BatchNorm.")
    # ap.add_argument("-mum", "--upmode", default='upsample', help="(Only for Model ID 0, 11) UpMode for model ID 0 and 11: [upconv, upsample], for model ID 9: [convtrans, <interp algo>]")
    # ap.add_argument("-mdp", "--mdepth", type=int, default=3, help="(Only for Model ID 0, 6, 11) Depth of the Model.")
    # ap.add_argument("-d", "--dropprob", type=float, default=0.0, help="(Only for Model ID 0, 6, 11) Dropout Probability.")
    # ap.add_argument("-mslvl", "--msslevel", type=int, default=2, help="(Only for Model ID 11) Depth of the Model.")
    # ap.add_argument("-msltn", "--msslatent", type=int, default=1, help="(Only for Model ID 11) Use the latent as one of the MSS level.")
    # ap.add_argument("-msup", "--mssup", default="trilinear", help="(Only for Model ID 11) Interpolation to use on the MSS levels.")
    # ap.add_argument("-msinb4", "--mssinterpb4", type=int, default=0, help="(Only for Model ID 11) Apply Interpolation before applying conv for the MSS levels. If False, interp will be applied after conv.")
    # ap.add_argument("-f", "--nfeatures", type=int, default=64, help="(Not for DenseNet) N Starting Features of the Network.")
    # ap.add_argument("-lid", "--lossid", type=int, default=0, help="Loss ID."+str(lossIDs))
    # ap.add_argument("-plt", "--plosstyp", default="L1", help="(Only for Loss ID 0) Perceptual Loss Type.")
    # ap.add_argument("-pll", "--plosslvl", type=int, default=3, help="(Only for Loss ID 0) Perceptual Loss Level.")
    # ap.add_argument("-lrd", "--lrdecrate", type=int, default=1, help="(To be used for Fine-Tuning) Factor by which lr will be divided to find the actual lr. Set it to 1 if not desired")
    # ap.add_argument("-ft", "--finetune", type=int, default=0, help="Is it a Fine-tuning traing or not (main-train).")
    # ap.add_argument("-ftep", "--fteprt", type=float, default=0.00, help="(To be used for Fine-Tuning) Fine-Tune Epoch Rate.")
    # ap.add_argument("-ftit", "--ftitrt", type=float, default=0.10, help="(To be used for Fine-Tuning, if fteprt is None) Fine-Tune Iteration Rate.")

    # ap.add_argument("-tls", "--tnnlslc", type=int, default=2, help="Solo per ThisNewNet. loss_slice_count. Default 2")    
    # ap.add_argument("-tli", "--tnnlinp", type=int, default=1, help="Solo per ThisNewNet. loss_inplane. Default 1")

    # #WnB related params
    # ap.add_argument("-wnb", "--wnbactive", type=bool, default=True, help="WandB: Whether to use or not")
    # ap.add_argument("-wnbp", "--wnbproject", default='SuperResMRI', help="WandB: Name of the project")
    # ap.add_argument("-wnbe", "--wnbentity", default='mickchimp', help="WandB: Name of the entity")
    # ap.add_argument("-wnbg", "--wnbgroup", default='dynDualChnFullVol', help="WandB: Name of the group")
    # ap.add_argument("-wnbpf", "--wnbprefix", default='', help="WandB: Prefix for TrainID")
    # ap.add_argument("-wnbml", "--wnbmodellog", default='all', help="WandB: While watching the model, what to save: gradients, parameters, all, None")
    # ap.add_argument("-wnbmf", "--wnbmodelfreq", type=int, default=100, help="WandB: The number of steps between logging gradients")

    return ap.parse_args()

args = parseARGS()
# os.environ["TMPDIR"] = "/scratch/schatter/tmp"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.set_num_threads(1)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__" :
    args.scalefact = tuple(map(int, args.scalefact.replace('(','').replace(')','').split(',')))  
    args.homepath = os.path.expanduser("~/Documents")
    if args.patchsize:
        args.patchsize = tuple(map(int, args.patchsize.replace('(','').replace(')','').split(',')))
    if args.patchstride:
        args.patchstride = tuple(map(int, args.patchstride.replace('(','').replace(')','').split(',')))
    if args.inshape:
        args.inshape = tuple(map(int, args.inshape.replace('(','').replace(')','').split(',')))
    # args.chkpoint = os.path.join(args.outpath, args.outfolder, args.modelname, args.modelname)
    # if args.modelbest:
    #     print('best model testing')
    #     args.chkpoint += "_" + args.beststring + ".pth.tar"
    # else:
    #     args.chkpoint += ".pth.tar"    

    # if args.patchstride:
    #     args.modelname += "_infstr" + "c".join(list(map(str, args.patchstride)))
    #     args.modelname = args.modelname.replace(args.usfolder+"_", "")

    # print("Testing: "+args.modelname)
    # if args.modelid == 2:
    #     SRCNN3D = SRCNN3Dv2
    # elif args.modelid == 3:
    #     SRCNN3D = SRCNN3Dv3

    # if args.medianloss:
    #     loss_reducer = statistics.median
    # else:
    #     loss_reducer = statistics.mean

    dir_path = args.dataset + args.usfolder+ '/' + args.us + '/'
    label_dir_path = args.dataset + args.hrfolder + '/'

    # log_path = os.path.join(args.dataset, args.outfolder, 'TBLogs', args.modelname)
    save_path = os.path.join(args.outpath, args.outfolder, args.modelname, args.outtype)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # tb_writer = SummaryWriter(log_dir = log_path)
    os.makedirs(save_path, exist_ok=True)
    # logname = os.path.join(args.homepath, 'testlog_'+args.modelname+'.txt')

    # logging.basicConfig(filename=logname,
    #                         filemode='a',
    #                         format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #                         datefmt='%H:%M:%S',
    #                         level=logging.DEBUG)
                            
    # # transforms = [tio.transforms.RescaleIntensity((0, 1))]
    # transforms = []

    testDS = SRDataset(logger=logging, patch_size=args.patchsize[0] if args.patchsize else -1, dir_path=dir_path, label_dir_path=label_dir_path, #TODO: implement non-iso patch-size, now only using the first element
                        stride_depth=args.patchstride[2], stride_length=args.patchstride[0], stride_width=args.patchstride[1], Size=None, fly_under_percent=None, #TODO: implement fly_under_percent, if needed 
                        patch_size_us=None, return_coords=True, pad_patch=False, pre_interpolate=args.preint, norm_data=args.prenorm, pre_load=True, noncumulative=True) #TODO implement patch_size_us if required - patch_size//scaling_factor
    test_loader = torch.utils.data.DataLoader(testDS, batch_size=args.batchsize,shuffle=False, num_workers=args.nworkers, pin_memory=True)

    # model_scale_factor=tuple(np.roll(args.scalefact,shift=1))

    # if args.modelid == 0:
    #     model = UNet(in_channels=args.inchannel, n_classes=args.outchannel, depth=args.mdepth, wf=round(math.log(args.nfeatures,2)), batch_norm=args.batchnorm, up_mode=args.upmode, dropout=bool(args.dropprob))
    # elif (args.modelid == 1) or (args.modelid == 2) or (args.modelid == 3):
    #     sys.exit("SRCNN3D is not ready for different numbers of input and output channel")
    #     model = SRCNN3D(n_channels=args.nchannel, scale_factor=model_scale_factor, num_features=args.nfeatures)    
    # elif (args.modelid == 4) or (args.modelid == 5):
    #     model = UNetVSeg(in_ch=args.inchannel, out_ch=args.outchannel, n1=args.nfeatures)   
    # elif args.modelid == 6:
    #     model = DenseNet(model_depth=args.mdepth, n_input_channels=args.inchannel, num_classes=args.outchannel, drop_rate=args.dropprob)
    # elif (args.modelid == 7) or (args.modelid == 8):
    #     model = ThisNewNet(in_channels=args.inchannel, n_classes=args.outchannel, depth=args.mdepth, batch_norm=args.batchnorm, up_mode=args.upmode, dropout=bool(args.dropprob), 
    #                         scale_factor=model_scale_factor, num_features=args.nfeatures, sliceup_first=True if args.modelid==8 else False, 
    #                         loss_slice_count=args.tnnlslc, loss_inplane=args.tnnlinp)
    # elif args.modelid == 9:   
    #     model=ResNet(in_channels=args.inchannel, out_channels=args.outchannel, res_blocks=4, starting_nfeatures=args.nfeatures, updown_blocks=2, is_relu_leaky=True,  #TODO: put all params as args
    #                 do_batchnorm=args.batchnorm, res_drop_prob=0.2, is_replicatepad=0, out_act="sigmoid", forwardV=0, upinterp_algo='convtrans' if args.upmode == "upconv" else "trilinear", post_interp_convtrans=True, is3D=True)
    # elif args.modelid == 10:
    #     model=ShuffleUNet(in_ch=args.inchannel, num_features=args.nfeatures, out_ch=args.outchannel)
    # elif args.modelid == 11:
    #     model = UNetMSS(in_channels=args.inchannel, n_classes=args.outchannel, depth=args.mdepth, wf=round(math.log(args.nfeatures,2)), 
    #                     batch_norm=args.batchnorm, up_mode=args.upmode, dropout=bool(args.dropprob),
    #                     mss_level=args.msslevel, mss_fromlatent=args.msslatent, mss_up=args.mssup, mss_interpb4=args.mssinterpb4)
    # else:
    #     sys.exit("Invalid Model ID")

    # if args.modelid == 5:
    #     IsDeepSup = True
    # else:
    #     IsDeepSup = False

    # if args.profile:
    #     dummy = torch.randn(args.batchsize, args.inchannel, *args.inshape)
    #     with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=True) as prof:
    #         model(dummy)
    #         prof.export_chrome_trace(os.path.join(save_path, 'model_trace'))
    # model.to(device)

    # chk = torch.load(args.chkpoint, map_location=device)
    # model.load_state_dict(chk['state_dict'])
    # trained_epoch = chk['epoch'] 
    # model.eval()

    saver = ResSaver(os.path.join(save_path, "Results"), save_inp=True, do_norm=args.prenorm)

    markers = {}
    inputs = {}
    results = {}
    targets = {}

    if not args.wnbactive:
        os.environ["WANDB_MODE"] = "dryrun"

    with torch.no_grad():
        runningSSIM = []
        test_ssim = []
        test_metrics = []        

        for b, (lr_imgs, hr_imgs, start_coords, files, shapes, pad) in enumerate(tqdm(test_loader)):
            lr_imgs = lr_imgs[:,1,...].unsqueeze(1).contiguous().to(device, non_blocking=True)  # (batch_size (N), 3, 24, 24), imagenet-normed
            hr_imgs = hr_imgs.contiguous()#.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]
            pad = pad.numpy()
            
            lr_imgs = F.interpolate(lr_imgs, size=hr_imgs.shape[2:], mode='trilinear') 
            tmp_in = lr_imgs.cpu().detach()#.numpy()
            tmp_tar = hr_imgs#.numpy()

            for i in range(hr_imgs.shape[0]):
                if bool(args.patchsize) and args.patchsize[0] != -1: #TODO: implement non-iso patch-size, now only using the first element
                    if files[i] not in results:
                        markers[files[i]] = np.zeros(shapes[i][0].numpy())
                        inputs[files[i]] = np.zeros(shapes[i][0].numpy())
                        results[files[i]] = np.zeros(shapes[i][0].numpy())
                        targets[files[i]] = np.zeros(shapes[i][0].numpy())

                    (startIndex_depth, startIndex_length, startIndex_width) = start_coords[i][0].numpy() #because of moveaxis, l,w,d has become d,l,w
                    if pad[i].any():
                        tin = F.pad(tmp_in[i].unsqueeze(0), tuple(-pad[i])).squeeze().numpy()
                        ttar = F.pad(tmp_tar[i].unsqueeze(0), tuple(-pad[i])).squeeze().numpy()
                    else:
                        tin = tmp_in[i].squeeze().numpy()
                        ttar = tmp_tar[i].squeeze().numpy()

                    tin = tin[1,...] #TODO make it configurable. Currently its prevTPPatch, patch
                    markers[files[i]][startIndex_length:startIndex_length+args.patchsize[0], startIndex_width:startIndex_width+args.patchsize[1], startIndex_depth:startIndex_depth+args.patchsize[2]] += 1
                    inputs[files[i]][startIndex_length:startIndex_length+args.patchsize[0], startIndex_width:startIndex_width+args.patchsize[1], startIndex_depth:startIndex_depth+args.patchsize[2]] += np.moveaxis(tin, 0, -1) 
                    targets[files[i]][startIndex_length:startIndex_length+args.patchsize[0], startIndex_width:startIndex_width+args.patchsize[1], startIndex_depth:startIndex_depth+args.patchsize[2]] += np.moveaxis(ttar, 0, -1) 
                else:                    
                    inputs[files[i]] = np.moveaxis(tmp_in[i,0,...].squeeze().numpy(), 0, -1)  #TODO make it configurable. Currently its prevTPPatch, patch
                    targets[files[i]] = np.moveaxis(tmp_tar[i,0,...].squeeze().numpy(), 0, -1)

        if bool(args.patchsize) and args.patchsize[0] != -1:
            for f in inputs.keys():
                inputs[f] = np.divide(inputs[f], markers[f])
                results[f] = np.divide(results[f], markers[f])
                targets[f] = np.divide(targets[f], markers[f])

        for i, filename in enumerate(results.keys()):
            out = results[filename]
            inp = inputs[filename]
            gt = targets[filename]
            metrics = saver.CalcNSave(out, inp, gt, filename, already_numpy=True)

            if metrics is not None:
                metrics['file'] = filename
                test_metrics.append(metrics)

                ssim = round(metrics['SSIMOut'],4)
                test_ssim.append(ssim)
                runningSSIM.append(ssim)
        
        if len(test_metrics) > 0:
            df = pd.DataFrame.from_dict(test_metrics)
            df.to_csv(os.path.join(save_path, 'Results.csv'), index=False)
