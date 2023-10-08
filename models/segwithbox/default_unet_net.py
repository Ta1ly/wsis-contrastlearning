from models.segwithbox.residualunet import ResidualUNet,ResidualUNetBackbone
from models.segwithbox.enet import ENet

__all__ = ["unet_residual", "enet", "unet_residual_backbone"]

def unet_residual(input_dim, num_classes, softmax, channels_in=32):
    model = ResidualUNet(input_dim, num_classes, softmax, channels_in)
    return model

def enet(input_dim, num_classes, softmax, channels_in=16):
    model = ENet(input_dim, num_classes, softmax, channels_in)
    return model

def unet_residual_backbone(input_dim, channels_in=32):
    model = ResidualUNetBackbone(input_dim, channels_in)
    return model