import datetime
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt
import warnings 
warnings.filterwarnings("ignore")

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from datasets.transform import random_transform_generator
from datasets.image import random_visual_effect_generator
from datasets.samplers import PatientSampler
from models.segwithbox.unetwithcam import UNetWithCAM
from models.batch_image import BatchImage
from models.segwithbox.default_unet_net import *
from utils.losses import *
from utils.ccam_losses import *
from utils.early_stop import EarlyStopping
from utils import config, utils
from utils.promise_utils import get_promise
from utils.engine import train_one_epoch, validate_loss
import utils.transforms as T
from _C_promise import _C


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=0, type=int,
                        help='the index of experiments')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print(args)

    n_exp = args.n_exp
    cfg = {'train_params': {'patience': 8, 'lr': 1e-4, 'batch_size': 16}, 
           'data_params': {'workers': 16}}
    
    if n_exp==0: # full supervision
        configs = [
            {'net_params': {'softmax': False, 'losses': 
                                [('CrossEntropyLoss', {'mode':'all'}, 1)]},
                   'save_params': {'experiment_name': 'residual_all_fs'}}
            ]
    
    elif n_exp == 1:
        configs = [{'net_params': {'model_name':'UNetWithCAM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'test_resunet_cam'},
                   'data_params': {'margin':1, 'random_margin':False}}]
    
    elif n_exp == 2:
        configs = [{'net_params': {'model_name':'UNetWithCAM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'test_resunet_cam_margin_2'},
                   'data_params': {'margin':2, 'random_margin':False}},
                   {'net_params': {'model_name':'UNetWithCAM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'test_resunet_cam_margin_4'},
                   'data_params': {'margin':4, 'random_margin':False}}
                   ,{'net_params': {'model_name':'UNetWithCAM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'test_resunet_cam_margin_8'},
                   'data_params': {'margin':8, 'random_margin':False}}]
    
    elif n_exp == 3:
        configs = [{'net_params': {'model_name':'UNetWithCAM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'mil_focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'mil_resunet_cam_margin_1'},
                   'data_params': {'margin':1, 'random_margin':False}},
                   {'net_params': {'model_name':'UNetWithCAM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'mil_focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'mil_resunet_cam_margin_2'},
                   'data_params': {'margin':2, 'random_margin':False}},
                   {'net_params': {'model_name':'UNetWithCAM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'mil_focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'mil_resunet_cam_margin_4'},
                   'data_params': {'margin':4, 'random_margin':False}}
                   ,{'net_params': {'model_name':'UNetWithCAM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'mil_focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'mil_resunet_cam_margin_8'},
                   'data_params': {'margin':8, 'random_margin':False}}]
        
    elif n_exp == 4:
        configs = [{'net_params': {'model_name':'UNetWithFSRM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'mil_focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'mil_resunet_fsrm_margin_1'},
                   'data_params': {'margin':1, 'random_margin':False}},
                   {'net_params': {'model_name':'UNetWithFSRM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'mil_focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'mil_resunet_fsrm_margin_2'},
                   'data_params': {'margin':2, 'random_margin':False}},
                   {'net_params': {'model_name':'UNetWithFSRM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'mil_focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'mil_resunet_fsrm_margin_4'},
                   'data_params': {'margin':4, 'random_margin':False}}
                   ,{'net_params': {'model_name':'UNetWithFSRM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'mil_focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'mil_resunet_fsrm_margin_8'},
                   'data_params': {'margin':8, 'random_margin':False}}]
    
    # for test    
    elif n_exp == -1:
        configs = [{'net_params': {'model_name':'UNetWithCAM','backbone':'unet_residual_backbone','softmax': False, 'losses': 
                                [('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'mil_focal', 'method':'expsumr', 'gpower':8},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10), ('ContrastLoss',{'alpha':0.25, 'metric':'cos'}, 1)]},
                   'save_params': {'experiment_name': 'test_model'},
                   'data_params': {'margin':1, 'random_margin':False}},
                   ]
    
    _C = config.config_updates(_C, cfg)
    _C_array = []
    
    for c in configs:
        _C_array.append(config.config_updates(_C, c))

    

    for _C_used in _C_array:
        assert _C_used['save_params']['experiment_name'] is not None, "experiment_name has to be set"

        train_params       = _C_used['train_params']
        data_params        = _C_used['data_params']
        net_params         = _C_used['net_params']
        dataset_params     = _C_used['dataset']
        save_params        = _C_used['save_params']
        data_visual_aug    = data_params['data_visual_aug']
        data_transform_aug = data_params['data_transform_aug']


        output_dir = os.path.join(save_params['dir_save'],save_params['experiment_name'])
        os.makedirs(output_dir, exist_ok=True)
        if not train_params['test_only']:
            config.save_config_file(os.path.join(output_dir,'config.yaml'), _C_used)
            print("saving files to {:s}".format(output_dir))

        device = torch.device(_C_used['device'])

        def get_transform():
            transforms = []
            transforms.append(T.ToTensor())
            transforms.append(T.Normalizer(mode=data_params['normalizer_mode']))
            return T.Compose(transforms)

        if data_transform_aug['aug']:
            transform_generator = random_transform_generator(
                min_rotation=data_transform_aug['min_rotation'],
                max_rotation=data_transform_aug['max_rotation'],
                min_translation=data_transform_aug['min_translation'],
                max_translation=data_transform_aug['max_translation'],
                min_shear=data_transform_aug['min_shear'],
                max_shear=data_transform_aug['max_shear'],
                min_scaling=data_transform_aug['min_scaling'],
                max_scaling=data_transform_aug['max_scaling'],
                flip_x_chance=data_transform_aug['flip_x_chance'],
                flip_y_chance=data_transform_aug['flip_y_chance'],
                )
        else:
            transform_generator = None
        if data_visual_aug['aug']:
            visual_effect_generator = random_visual_effect_generator(
                contrast_range=data_visual_aug['contrast_range'],
                brightness_range=data_visual_aug['brightness_range'],
                hue_range=data_visual_aug['hue_range'],
                saturation_range=data_visual_aug['saturation_range']
                )
        else:
            visual_effect_generator = None
        print('---data augmentation---')
        print('transform: ',transform_generator)
        print('visual: ',visual_effect_generator)

        # Data loading code
        print("Loading data")
        # print('0', data_params['random_margin'])
        dataset      = get_promise(root=dataset_params['root_path'], 
                                   image_folder=dataset_params['train_path'][0], 
                                   gt_folder=dataset_params['train_path'][1], 
                                   transforms=get_transform(),
                                   transform_generator=transform_generator,
                                   visual_effect_generator=visual_effect_generator, margin=data_params['margin'], random_margin=data_params['random_margin'])
        dataset_test = get_promise(root=dataset_params['root_path'], 
                                   image_folder=dataset_params['valid_path'][0], 
                                   gt_folder=dataset_params['valid_path'][1], 
                                   transforms=get_transform(),
                                   transform_generator=None, visual_effect_generator=None)
        
        print("Creating data loaders")
        train_sampler = torch.utils.data.RandomSampler(dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, train_params['batch_size'], drop_last=True)

        # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_patient_sampler = PatientSampler(dataset_test, dataset_params['grp_regex'], shuffle=False)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=data_params['workers'],
            collate_fn=utils.collate_fn, pin_memory=True)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            batch_sampler=test_patient_sampler, num_workers=data_params['workers'],
            collate_fn=utils.collate_fn, pin_memory=True)

        print("Creating model with parameters: {}".format(net_params))
        losses, loss_weights = [], []
        for loss in net_params['losses']:
            losses.append(eval(loss[0])(**loss[1]))
            loss_weights.append(loss[2])
        batch_image = BatchImage(size_divisible=32)
        batch_image.to(device)
    
        if net_params["backbone"] == "unet_residual_backbone":
            backbone = eval(net_params['backbone'])(net_params['input_dim'])
        else:
            raise ValueError("backbone not implement yet")
    
        if net_params["model_name"] == "UNetWithFSRM":
            model = UNetWithFSRM(backbone, softmax=net_params['softmax'], num_classes = net_params['seg_num_classes'])
        elif net_params["model_name"] == "UNetWithCAM":
            model = UNetWithCAM(backbone, softmax=net_params['softmax'], num_classes = net_params['seg_num_classes'])
        else:
            raise ValueError(f"{net_params['model_name']} model not implement yet")
        
        model.to(device)
        

        params = [p for p in model.parameters() if p.requires_grad]
        if train_params['optimizer']=='SGD':
            optimizer = torch.optim.SGD(params, lr=train_params['lr'], 
                                        momentum=train_params['momentum'], weight_decay=train_params['weight_decay'])
        elif train_params['optimizer']=='Adam':
            optimizer = torch.optim.Adam(params, lr=train_params['lr'], betas=train_params['betas'])
            
        elif train_params['optimizer']=='AdamW':
            optimizer = torch.optim.AdamW(params, lr=train_params['lr'], betas=train_params['betas'], weight_decay=train_params['weight_decay'])

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                  factor=train_params['factor'], 
                                                                  patience=train_params['patience'])
        
        early_stop = EarlyStopping(patience=train_params['patience'])

        if train_params['resume']:
            print('resuming model {}'.format(train_params['resume']))
            checkpoint = torch.load(os.path.join(output_dir, train_params['resume']), map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            train_params['start_epoch'] = checkpoint['epoch'] + 1

        torch.autograd.set_detect_anomaly(True)
        model.training = True
        if train_params['test_only']:
            val_metric_logger = validate_loss(model, batch_image, data_loader_test, device)
        else:
            print("Start training")
            start_time = time.time()
            summary = {'epoch':[]}
            for epoch in range(train_params['start_epoch'], train_params['epochs']):
                if args.distributed:
                    train_sampler.set_epoch(epoch)

                metric_logger = train_one_epoch(model, batch_image, losses, loss_weights, optimizer, data_loader, device, epoch, 
                                train_params['clipnorm'], train_params['print_freq'])
                
                
                if output_dir:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'args': args,
                        'epoch': epoch},
                        os.path.join(output_dir, 'model_{:02d}.pth'.format(epoch)))

                # evaluate after every epoch
                val_metric_logger = validate_loss(model, batch_image, losses, loss_weights, data_loader_test, device)

                # collect the results and save 
                summary['epoch'].append(epoch)
                for name, meter in metric_logger.meters.items():
                    if name=='lr':
                        v = meter.global_avg
                    else:
                        v = float(np.around(meter.global_avg,4))
                    if epoch==0:
                        summary[name] = [v]
                    else:
                        summary[name].append(v)
                for name, meter in val_metric_logger.meters.items():
                    v = float(np.around(meter.global_avg,4))
                    if epoch==0:
                        summary[name] = [v]
                    else:
                        summary[name].append(v)
                summary_save = pd.DataFrame(summary)
                summary_save.to_csv(os.path.join(output_dir,'summary.csv'), index=False)

                # update lr scheduler
                val_loss = val_metric_logger.meters['val_loss'].global_avg
                lr_scheduler.step(val_loss)

                # # early stop check
                # if early_stop.step(val_loss):
                #     print('Early stop at epoch = {}'.format(epoch))
                #     break

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))

            ## plot training and validation loss
            plt.figure()
            plt.plot(summary_save['epoch'],summary_save['loss'],'-ro', label='train')
            plt.plot(summary_save['epoch'],summary_save['val_loss'],'-g+', label='valid')
            plt.legend(loc=0)
            plt.savefig(os.path.join(output_dir,'loss.jpg'))
