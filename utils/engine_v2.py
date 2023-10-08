import math
import sys
import time
import numpy as np
import torch
from utils import utils
from torch.jit.annotations import Tuple, List, Dict, Optional

def check_for_degenerate_bboxes(targets):
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError("All bounding boxes should have positive height and width."
                                    " Found invalid box {} for target at index {}."
                                    .format(degen_bb, target_idx))


def prepare_kwargs_opt(images, targets, seg_preds, fg_feats, bg_feats, device):
    image_shape = images.tensors.shape
    # print("image shape:", image_shape, "seg_preds shape: ",seg_preds.shape, "ccam shape:" , ccam.shape,  "feature shape:", fg_feats.shape, "len: ", len(seg_preds))
    dtype  = seg_preds.dtype
    
    all_labels = [t['labels'] for t in targets]
    ytrue = torch.stack([t['masks'] for t in targets],dim=0).long()
    label_unique = torch.unique(torch.cat(all_labels, dim=0))
    
    
    # for nb_level in range(len(seg_preds)):
    preds = seg_preds
    stride = image_shape[-1]/preds.shape[-1]

    mask = preds.new_full(preds.shape,0,device=preds.device)
    crop_boxes = []
    gt_boxes   = []
    for n_img, target in enumerate(targets):
        boxes = torch.round(target['boxes']/stride).type(torch.int32)
        labels = target['labels']
        for n in range(len(labels)):
            box = boxes[n,:]
            c   = labels[n]#.item()
            # print("####", mask.shape, n_img,c,box[1],box[3]+1,box[0],box[2]+1)
            mask[n_img,c,box[1]:box[3]+1,box[0]:box[2]+1] = 1

            height, width = (box[2]-box[0]+1)/2.0, (box[3]-box[1]+1)/2.0
            r  = torch.sqrt(height**2+width**2)
            cx = (box[2]+box[0]+1)//2
            cy = (box[3]+box[1]+1)//2
            # print('//// box ////',box, cx, cy, r)
            crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
            gt_boxes.append(torch.tensor([n_img, c, box[0], box[1], box[2], box[3]], dtype=torch.int32, device=device))
    if len(crop_boxes)==0:
        crop_boxes = torch.empty((0,5), device=device)
    else:
        crop_boxes = torch.stack(crop_boxes, dim=0)
    if len(gt_boxes)==0:
        gt_boxes = torch.empty((0,6), device=device)
    else:
        gt_boxes = torch.stack(gt_boxes, dim=0) 

    # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
    assert crop_boxes.shape[0]==gt_boxes.shape[0]
    return {'ypred':preds, 'ytrue':ytrue, 'mask':mask, 'gt_boxes':gt_boxes, 'crop_boxes':crop_boxes, 'fg_feats':fg_feats, 'bg_feats':bg_feats}
        


def train_one_epoch(model, batch_image, losses_func, loss_weights, optimizer, data_loader, device, epoch, clipnorm=0.001, print_freq=20):
    time.sleep(2)  # Prevent possible deadlock during epoch transition
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        if targets is None:
            raise ValueError("In training mode, targets should be passed")
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    raise ValueError("Expected target boxes to be a tensor"
                                        "of shape [N, 4], got {:}.".format(
                                            boxes.shape))
            else:
                raise ValueError("Expected target boxes to be of type "
                                    "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = batch_image(images, targets)
        bbox_prior_mask = bbox_prior_mask_generator(images.tensors, targets) # generate bbox mask
        if torch.isnan(images.tensors).sum()>0:
            print('image is nan ..............')
        if torch.isinf(images.tensors).sum()>0:
            print('image is inf ..............')   
                                               
        # check_for_degenerate_bboxes(targets)
        
        [seg_preds, fg_feats, bg_feats, ccam] = model(images, targets)

        ########################################################################
        ##############           START CALCU  LOSS               ###############
        ########################################################################
        seg_losses = {}
        kwargs_opt = prepare_kwargs_opt(images, targets, seg_preds, fg_feats, bg_feats, device)
        # mil_loss = loss_func[0]() * loss_w[0]
        # pairwise_loss = loss_func[1]() * loss_w[1]
        # contrast_loss = loss_func[2](fg_feats, bg_feats) * loss_w[2]
        
        
        for loss_func, loss_w in zip(losses_func, loss_weights):
            loss_keys = loss_func.__call__.__code__.co_varnames
            # print("loss keys ", loss_keys)
            loss_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in loss_keys}
            loss_v = loss_func(**loss_params)*loss_w
            # print(loss_v.shape, loss_v.ndim)
            key_prefix = type(loss_func).__name__+'/'
            if loss_v.ndim > 0:
                loss_v = {key_prefix+str(n):loss_v[n] for n in range(len(loss_v))}
            else:
                loss_v = {key_prefix:loss_v}
            seg_losses.update(loss_v)
        
        losses = sum(loss for loss in seg_losses.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(seg_losses)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        ########################################################################
        ##############             END CALCU LOSS                ###############
        ########################################################################


        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if clipnorm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipnorm)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def validate_loss(model, batch_image, losses_func, loss_weights, data_loader, device):
    n_threads = torch.get_num_threads()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation: '

    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    import pandas as pd
    loss_summary = []
    for images, targets in metric_logger.log_every(data_loader, print_freq=10e5, header=header, training=False):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        images, targets = batch_image(images, targets)
        bbox_prior_mask = bbox_prior_mask_generator(images.tensors, targets) # generate bbox mask
        
        [seg_preds, fg_feats, bg_feats, ccam] = model(images, targets)
        
        kwargs_opt = prepare_kwargs_opt(images, targets, seg_preds, fg_feats, bg_feats, device)
        
        seg_losses = {}
        
        for loss_func, loss_w in zip(losses_func, loss_weights):
            loss_keys = loss_func.__call__.__code__.co_varnames
            # print("loss keys ", loss_keys)
            loss_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in loss_keys}
            loss_v = loss_func(**loss_params)*loss_w
            # print(loss_v.shape, loss_v.ndim)
            key_prefix = type(loss_func).__name__+'/'
            if loss_v.ndim > 0:
                loss_v = {key_prefix+str(n):loss_v[n] for n in range(len(loss_v))}
            else:
                loss_v = {key_prefix:loss_v}
            seg_losses.update(loss_v)
        
        losses = sum(loss for loss in seg_losses.values())


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(seg_losses)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_dict_reduced = dict(('val_'+k,f(v) if hasattr(v,'keys') else v) for k,v in loss_dict_reduced.items())

        loss_value = losses_reduced.item()
        metric_logger.update(val_loss=losses_reduced, **loss_dict_reduced)

        loss_reduced = dict((k,f(v) if hasattr(v,'keys') else v.item()) for k,v in loss_dict_reduced.items())
        loss_reduced.update(dict(gt=targets[0]["boxes"].shape[0]))
        loss_summary.append(loss_reduced)

    loss_summary = pd.DataFrame(loss_summary)
    loss_summary.to_csv("val_image_summary.csv", index=False)

    torch.set_num_threads(n_threads)
    
    return metric_logger

