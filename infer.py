import os, sys
import warnings 
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from datasets.samplers import PatientSampler
from models.segwithbox.unetwithcam import UNetWithCAM
from models.batch_image import BatchImage
from models.segwithbox.default_unet_net import *
from utils.losses import *

from utils.ccam_losses import *
from utils import config, utils, pd_utils 
from utils.promise_utils import get_promise
import utils.transforms as T

@torch.no_grad()
def evaluate(epoch, model, batch_image, data_loader, image_names, device, threshold, save_detection=None, smooth=1e-10):
    file_2d = os.path.join(save_detection,'dice_2d.xlsx')
    file_3d = os.path.join(save_detection,'dice_3d.xlsx')
    torch.set_num_threads(1)
    model.eval()

    nn = 0
    dice_2d, dice_3d = {k:[] for k in range(len(threshold))}, {k:[] for k in range(len(threshold))}
    for images, targets in data_loader:
        nn = nn + 1
        # print("{}/{}".format(nn,len(data_loader))) 

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gt = torch.stack([t["masks"] for t in targets], dim=0)
        gt = gt.bool()
        print(gt.shape)
        # _, outputs = model(images, targets)
        
        images, targets = batch_image(images, targets)
        # bbox_prior_mask = bbox_prior_mask_generator(images.tensors, targets) # generate bbox mask
        [seg_preds, fg_feats, bg_feats, ccam] = model(images, targets)
        outputs = seg_preds
        print(outputs.shape)
        
        
        # for out in outputs:
        for n_th,th in enumerate(threshold):
            pred = outputs>th
            intersect = pred&gt
            v_dice_2d = (2*torch.sum(intersect,dim=(1,2,3))+smooth)/(torch.sum(pred,dim=(1,2,3))+torch.sum(gt,dim=(1,2,3))+smooth)
            v_dice_3d = (2*torch.sum(intersect)+smooth)/(torch.sum(pred)+torch.sum(gt)+smooth)
            dice_2d[n_th].append(v_dice_2d.cpu().numpy())
            dice_3d[n_th].append(v_dice_3d.cpu().numpy())

    dice_2d = [np.hstack(dice_2d[key]) for key in dice_2d.keys()]
    dice_3d = [np.hstack(dice_3d[key]) for key in dice_3d.keys()]
    dice_2d = np.vstack(dice_2d).T
    dice_3d = np.vstack(dice_3d).T
    
    dice_2d = pd.DataFrame(data=dice_2d, columns=threshold)
    dice_3d = pd.DataFrame(data=dice_3d, columns=threshold)
    # print('dice 2d', dice_2d)
    # print('dice 3d', dice_3d)
    pd_utils.append_df_to_excel(file_2d, dice_2d, sheet_name=str(epoch), index=False)
    pd_utils.append_df_to_excel(file_3d, dice_3d, sheet_name=str(epoch), index=False)

    mean_2d = np.mean(dice_2d, axis=0)
    std_2d = np.std(dice_2d, axis=0)
    loc2 = np.argmax(mean_2d)
    mean_3d = np.mean(dice_3d, axis=0)
    std_3d = np.std(dice_3d, axis=0)
    loc3 = np.argmax(mean_3d)
    print('2d mean: {}({})'.format(mean_2d.tolist()[int(loc2)],std_2d.tolist()[int(loc2)]))
    print('3d mean: {}({})'.format(mean_3d.tolist()[int(loc3)],std_3d.tolist()[int(loc3)]))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=62, type=int,
                        help='the index of experiments')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print(args)

    n_exp = args.n_exp
    dir_save_root = os.path.join('results','promise')
    threshold = [0.001,0.005,0.01]+list(np.arange(0.05,0.9,0.05))
    ## fs
    if n_exp==0:
    	experiment_name = 'residual_all_fs'
    ## mil baseline
    elif n_exp==1:	
        experiment_name = 'residual_all_unary_pair'
    ## ablation study - generalized mil
    elif n_exp==2:
        experiment_name = 'residual_parallel_focal_40_10_unary_pair'
    elif n_exp==3:
        experiment_name = 'residual_parallel_focal_40_20_unary_pair'
    elif n_exp==4:
        experiment_name = 'residual_parallel_focal_60_30_unary_pair'
    ## ablation study - smooth max approximation
    elif n_exp==5:
        experiment_name = 'residual_all_unaryapprox_expsumr=4_pair'
    elif n_exp==6:
        experiment_name = 'residual_all_unaryapprox_expsumr=6_pair'
    elif n_exp==7:
        experiment_name = 'residual_all_unaryapprox_expsumr=8_pair'
    elif n_exp==8:
        experiment_name = 'residual_all_unaryapprox_explogs=4_pair'
    elif n_exp==9:
        experiment_name = 'residual_all_unaryapprox_explogs=6_pair'
    elif n_exp==10:
        experiment_name = 'residual_all_unaryapprox_explogs=8_pair'
    ## main results
    elif n_exp==11:
        experiment_name = 'residual_parallel_approx_focal_40_20_expsumr=4_unary_pair'
    elif n_exp==12:
        experiment_name = 'residual_parallel_approx_focal_40_20_expsumr=6_unary_pair'
    elif n_exp==13:
        experiment_name = 'residual_parallel_approx_focal_40_20_expsumr=8_unary_pair'
    elif n_exp==14:
        experiment_name = 'residual_parallel_approx_focal_40_20_explogs=4_unary_pair'
    elif n_exp==15:
        experiment_name = 'residual_parallel_approx_focal_40_20_explogs=6_unary_pair'
    elif n_exp==16:
        experiment_name = 'residual_parallel_approx_focal_40_20_explogs=8_unary_pair'
    
    
    ## random bbox 
    elif n_exp==17:
        experiment_name = 'residual_all_unary_pair_random_margin_1'
    elif n_exp==18:
        experiment_name = 'residual_all_unary_pair_random_margin_2'
    elif n_exp==19:
        experiment_name = 'residual_all_unary_pair_random_margin_4'
    elif n_exp==20:
        experiment_name = 'residual_all_unary_pair_random_margin_8'
    
    ## fix margin bbox 
    elif n_exp==21:
        experiment_name = 'residual_all_unary_pair_fix_margin_1'
    elif n_exp==22:
        experiment_name = 'residual_all_unary_pair_fix_margin_2'
    elif n_exp==23:
        experiment_name = 'residual_all_unary_pair_fix_margin_4'
    elif n_exp==24:
        experiment_name = 'residual_all_unary_pair_fix_margin_8'
        
    ## fix margin bbox + general mil
    elif n_exp==25:
        experiment_name = 'residual_parallel_focal_unary_pair_fix_margin_1'
    elif n_exp==26:
        experiment_name = 'residual_parallel_focal_unary_pair_fix_margin_2'
    elif n_exp==27:
        experiment_name = 'residual_parallel_focal_unary_pair_fix_margin_4'
    elif n_exp==28:
        experiment_name = 'residual_parallel_focal_unary_pair_fix_margin_8'

    ## fix margin bbox + tightness prior(explog)
    elif n_exp==29:
        experiment_name = 'residual_parallel_explogs_fix_margin_1'
    elif n_exp==30:
        experiment_name = 'residual_parallel_explogs_fix_margin_2'
    elif n_exp==31:
        experiment_name = 'residual_parallel_explogs_fix_margin_4'
    elif n_exp==32:
        experiment_name = 'residual_parallel_explogs_fix_margin_8'
    
    elif n_exp==33:
        experiment_name = 'test_resunet_cam'
        
    elif n_exp==34:
        experiment_name = 'test_resunet_cam_margin_2'
    elif n_exp==35:
        experiment_name = 'test_resunet_cam_margin_4'
    elif n_exp==36:
        experiment_name = 'test_resunet_cam_margin_8'
    elif n_exp == 37:
        experiment_name = 'mil_resunet_cam_margin_1'
    elif n_exp == 38:
        experiment_name = 'mil_resunet_cam_margin_2'
    elif n_exp == 39:
        experiment_name = 'mil_resunet_cam_margin_4'
    elif n_exp == 40:
        experiment_name = 'mil_resunet_cam_margin_8'


    print(experiment_name)
    output_dir = os.path.join(dir_save_root, experiment_name)
    _C = config.read_config_file(os.path.join(output_dir, 'config.yaml'))
    assert _C['save_params']['experiment_name']==experiment_name, "experiment_name is not right"
    cfg = {'data_params': {'workers': 4}}
    _C = config.config_updates(_C, cfg)

    train_params       = _C['train_params']
    data_params        = _C['data_params']
    net_params         = _C['net_params']
    dataset_params     = _C['dataset']
    save_params        = _C['save_params']

    device = torch.device(_C['device'])

    def get_transform():
        transforms = []
        transforms.append(T.ToTensor())
        transforms.append(T.Normalizer(mode=data_params['normalizer_mode']))
        return T.Compose(transforms)

    # Data loading code
    print("Loading data")
    dataset_test = get_promise(root=dataset_params['root_path'], 
                                   image_folder=dataset_params['valid_path'][0], 
                                   gt_folder=dataset_params['valid_path'][1], 
                                   transforms=get_transform(),
                                   transform_generator=None, visual_effect_generator=None)
    image_names = dataset_test.image_names
    
    print("Creating data loaders")
    test_patient_sampler = PatientSampler(dataset_test, dataset_params['grp_regex'], shuffle=False)

    data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            batch_sampler=test_patient_sampler, num_workers=data_params['workers'],
            collate_fn=utils.collate_fn, pin_memory=True)

    print("Creating model with parameters: {}".format(net_params))
    
    backbone = eval(net_params['backbone'])(net_params['input_dim'])
    # net = eval(net_params['model_name'])(net_params['input_dim'], net_params['seg_num_classes'],
    #                                          net_params['softmax'])
    losses, loss_weights = [], []
    for loss in net_params['losses']:
        losses.append(eval(loss[0])(**loss[1]))
        loss_weights.append(loss[2])
    if net_params["model_name"] == "UNetWithFSRM":
            model = UNetWithFSRM(backbone, softmax=net_params['softmax'], num_classes = net_params['seg_num_classes'])
    elif net_params["model_name"] == "UNetWithCAM":
        model = UNetWithCAM(backbone, softmax=net_params['softmax'], num_classes = net_params['seg_num_classes'])
    else:
        raise ValueError(f"{net_params['model_name']} model not implement yet")
    model.to(device)
    
    batch_image = BatchImage(size_divisible=32)
    file_2d = os.path.join(output_dir,'dice_2d.xlsx')
    file_3d = os.path.join(output_dir,'dice_3d.xlsx')
    if os.path.exists(file_2d):
        os.remove(file_2d)
    if os.path.exists(file_3d):
        os.remove(file_3d)
    for epoch in range(50):
        model_file = 'model_{:02d}'.format(epoch)
        print('loading model {}.pth'.format(model_file))
        checkpoint = torch.load(os.path.join(output_dir, model_file+'.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    
        print('start evaluating {} ...'.format(epoch))
        model.training = False
        evaluate(epoch, model, batch_image, data_loader_test, image_names=image_names, device=device, threshold=threshold, save_detection=output_dir)
        
    
    dice_2d_all = pd.read_excel(file_2d, sheet_name=None)
    dice_3d_all = pd.read_excel(file_3d, sheet_name=None)



    