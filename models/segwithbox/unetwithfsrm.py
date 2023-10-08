import copy
import torch
from torch import nn
from models.batch_image import BatchImage
from utils.losses import *
from torch.jit.annotations import Tuple, List, Dict, Optional
import numpy as np

__all__ = ['UNetWithFSRM']

class FSRM(nn.Module):
    def __init__(self, cin):
        super(FSRM, self).__init__()

        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)

    def forward(self, x, inference=False):
        N, C, H, W = x.size()
        if inference:
            ccam = self.bn_head(self.activation_head(x))
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))

        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam


class UNetWithFSRM(nn.Module):
    
    def __init__(self, backbone, softmax, cin=None, num_classes=2, feature_map_dim=32):

        super(UNetWithFSRM, self).__init__()        
        self.backbone = backbone
        self.num_classes = num_classes
        self.seg_head = nn.Conv2d(feature_map_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        if softmax:
            self.pred_func = nn.Softmax(dim=1)
        else:
            self.pred_func = nn.Sigmoid()
        self.ac_head = FSRM(feature_map_dim)

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        feats = self.backbone(images.tensors)
        # print(feats.shape)
        fg_feats, bg_feats, ccam = self.ac_head(feats, inference=not self.training)
        seg_preds = self.pred_func(self.seg_head(feats))

        return [seg_preds, fg_feats, bg_feats, ccam]
    
    def forward_fsrm(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        feats = self.backbone(images.tensors)
        # print(feats.shape)
        feats = feats 
        fg_feats, bg_feats, ccam = self.ac_head(feats, inference=not self.training)
        # fsrm_result = ccam * fg_feats
        return [fg_feats, bg_feats, ccam]
