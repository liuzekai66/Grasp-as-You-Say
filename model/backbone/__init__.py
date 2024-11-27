from .pointnet2 import Pointnet2Backbone
from .resnet import build_resnet_backbone
from .pointnet import PointNetEncoder
from .clip_sd import ClipCustom

def build_backbone(backbone_cfg):
    if backbone_cfg.name.lower() == "resnet":
        return build_resnet_backbone(backbone_cfg)
    elif backbone_cfg.name.lower() == "pointnet2":
        return Pointnet2Backbone(backbone_cfg)
    elif backbone_cfg.name.lower() == "pointnet":
        return PointNetEncoder(backbone_cfg)
    elif backbone_cfg.name.lower() == "clip_sd":
        return ClipCustom(backbone_cfg)
    else:
        raise NotImplementedError(f"No such backbone: {backbone_cfg.name}")
