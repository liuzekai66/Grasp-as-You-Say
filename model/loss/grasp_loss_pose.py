import torch
from torch import nn
import torch.nn.functional as F
from torch.functional import Tensor

import pytorch3d.structures
import pytorch3d.ops
import pytorch3d.transforms
from pytorch3d.loss import chamfer_distance

from typing import Dict, List, Tuple

from datasets import TaskDataset_Pose
from model.utils.hand_model import HandModel, contact_map_of_m_to_n
from utils.rotation_utils import Rot2Axisangle
from .matcher import Matcher

class GraspLossPose(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()

        self.hand_model = HandModel(loss_cfg.hand_model, loss_cfg.device)
        self.denorm_fuction = getattr(TaskDataset_Pose, f"denorm_by_{loss_cfg.norm_type}")
        self.norm_factor = getattr(TaskDataset_Pose, f"factor_{loss_cfg.norm_type[:6]}").to(loss_cfg.device)
        self.loss_weights = {k: v for k, v in loss_cfg.loss_weights.items() if v > 0}
        self.matcher = Matcher(weight_dict=loss_cfg.cost_weights)
        self.cfg = loss_cfg


    def forward(self, outputs, targets):
        if self.training:
            return self.forward_train(outputs, targets)
        else:
            return self.forward_test(outputs, targets)

    def forward_train(self, outputs, targets):
        outputs, targets = self.get_hand_model_pose(outputs, targets)
        outputs, targets = self.get_hand(outputs, targets)
        losses = {}
        for name, _ in self.loss_weights.items():
            if hasattr(self, f"get_{name}_loss"):
                m = getattr(self, f"get_{name}_loss")
                _loss_dict = m(outputs, targets)
                losses.update(_loss_dict)
            else:
                available_loss = [x[4:] for x in dir(self) if x.endswith("_loss") and not x.startswith("_")]
                raise NotImplementedError(f"Unable to calculate {name} loss. Available losses: {available_loss}")
        return losses

    def forward_test(self, outputs, targets):
        outputs, targets = self.get_hand_model_pose_test(outputs, targets)
        assignments = self.matcher(outputs, targets)  
        matched_preds, matched_targets = self.get_matched_by_assignment(outputs, targets, assignments)
        outputs["matched"] = matched_preds
        targets["matched"] = matched_targets
        targets["matched"]["obj_pc"] = targets["obj_pc"]
        outputs, targets = self.get_hand(outputs, targets)
        losses = {}
        for name, _ in self.loss_weights.items():
            if hasattr(self, f"get_{name}_loss"):
                m = getattr(self, f"get_{name}_loss")
                _loss_dict = m(outputs, targets)
                losses.update(_loss_dict)
            else:
                available_loss = [x[4:] for x in dir(self) if x.endswith("_loss") and not x.startswith("_")]
                raise NotImplementedError(f"Unable to calculate {name} loss. Available losses: {available_loss}")
        return losses, {"outputs": outputs, "targets": targets}
    
    def get_hand_model_pose(self, outputs, targets):
        targets["translation_norm"] = targets["norm_pose"][...,:3]
        targets["qpos_norm"] = targets["norm_pose"][...,3:25]
        targets["rotation"] = targets["norm_pose"][...,25:]
        
        outputs["translation_norm"] = outputs["pred_pose_norm"][...,:3]
        outputs["qpos_norm"] = outputs["pred_pose_norm"][...,3:25]
        outputs['rotation'] = outputs["pred_pose_norm"][...,25:]

        euler_angle = self.denorm_fuction(outputs['rotation'], self.norm_factor[3:6])
        axis_angle = torch.flip(
            pytorch3d.transforms.matrix_to_axis_angle(
            pytorch3d.transforms.euler_angles_to_matrix(euler_angle, "ZYX")),
            dims=[-1]
        )
        tranlation = self.denorm_fuction(outputs["translation_norm"], self.norm_factor[:3])
        qpos = self.denorm_fuction(outputs["qpos_norm"], self.norm_factor[6:])
        outputs["hand_model_pose"] = torch.cat([tranlation, axis_angle, qpos], dim=-1)     
        outputs = {"matched": outputs}
        targets = {"matched": targets}
        return outputs, targets

    def get_hand_model_pose_test(self, outputs, targets):
        if outputs["pred_pose_norm"].dim() < 3:
            outputs["pred_pose_norm"] = outputs["pred_pose_norm"].unsqueeze(1)
        try:
            if targets["norm_pose"].dim() < 3:
                targets["norm_pose"] = targets["norm_pose"].unsqueeze(1)
                targets["hand_model_pose"] = targets["hand_model_pose"].unsqueeze(1)
        except:
            pass
        outputs["translation_norm"] = outputs["pred_pose_norm"][...,:3]
        outputs["qpos_norm"] = outputs["pred_pose_norm"][...,3:25]
        outputs['rotation'] = outputs["pred_pose_norm"][...,25:]


        euler_angle = self.denorm_fuction(outputs['rotation'], self.norm_factor[3:6])
        axis_angle = torch.flip(
            pytorch3d.transforms.matrix_to_axis_angle(
            pytorch3d.transforms.euler_angles_to_matrix(euler_angle, "ZYX")),
            dims=[-1]
        )
        tranlation = self.denorm_fuction(outputs["translation_norm"], self.norm_factor[:3])
        qpos = self.denorm_fuction(outputs["qpos_norm"], self.norm_factor[6:])
        outputs["hand_model_pose"] = torch.cat([tranlation, axis_angle, qpos], dim=-1)    
        return outputs, targets

    def get_hand(self, outputs, targets):

        targets['hand'] = self.hand_model(
                            targets["matched"]["hand_model_pose"], 
                            targets["matched"]["obj_pc"],
                            with_meshes=True,
                            with_penetration=True,
                            with_surface_points=True,
                            with_penetration_keypoints=True)    
        outputs['hand'] = self.hand_model(
                            outputs["matched"]["hand_model_pose"], 
                            targets["matched"]["obj_pc"],
                            with_meshes=True,
                            with_penetration=True,
                            with_surface_points=True,
                            with_penetration_keypoints=True)


        outputs["rotation_type"] = self.cfg.rotation_type

        return outputs, targets

    def infer_norm_process_dict(self, outputs, targets):
        outputs, targets = self.get_hand_model_pose_test(outputs, targets)
        assignments = self.matcher(outputs, targets)  
        matched_preds, matched_targets = self.get_matched_by_assignment(outputs, targets, assignments)
        outputs["matched"] = matched_preds
        targets["matched"] = matched_targets
        targets["matched"]["obj_pc"] = targets["obj_pc"]
        outputs, targets = self.get_hand(outputs, targets)
        return outputs['hand'], targets['hand']


    def get_matched_by_assignment(
        self,
        predictions: Dict,
        targets: Dict,
        assignment: Dict,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            matched_preds: Dict[str, Tensor of size (K, D)]
            matched_targets: Dict[str, Tensor of size (K, D)]
        """
        per_query_gt_inds = assignment["per_query_gt_inds"] # (B, num_queries)
        query_matched_mask = assignment["query_matched_mask"]  # (B, num_queries)
        K = query_matched_mask.long().sum()  # K = number of matched
        B = query_matched_mask.size(0)
        matched_preds, matched_targets = {}, {}
        pred_target_match_key_map = {
            "pred_pose_norm": "norm_pose",
            "hand_model_pose": "hand_model_pose",
        }
        pose_slices = {
            "translation_norm": (0, 3),
            "qpos_norm": (3, 25),
            "rotation": (25, targets["norm_pose"][0].size(-1)),
        }
        for pred_key, target_key in pred_target_match_key_map.items():
            if pred_key not in predictions.keys():
                continue
            pred = predictions[pred_key]
            target = targets[target_key]
            matched_pred_buffer = []
            matched_target_buffer = []
            for i in range(B):
                _matched_pred, _matched_target = self._get_matched(
                    pred[i],
                    target[i],
                    per_query_gt_inds[i],
                    query_matched_mask[i],
                )
                matched_pred_buffer.append(_matched_pred)
                matched_target_buffer.append(_matched_target)
            matched_preds[pred_key] = torch.cat(matched_pred_buffer, dim=0)
            matched_targets[target_key] = torch.cat(matched_target_buffer, dim=0)
            if pred_key in pose_slices.keys():
                _s, _e = pose_slices[pred_key]
                matched_targets[target_key] = matched_targets[target_key][:, _s:_e]
            assert K == matched_preds[pred_key].size(0)
            assert K == matched_targets[target_key].size(0)
        return matched_preds, matched_targets
    
    def _get_matched(self, pred, gt, gt_inds, matched_mask) -> Tuple[Tensor, Tensor]:
        """
        Params:
        pred: A tensor of size (N, D)
        gt: A tensor of size (M, D)
        gt_inds: A tensor of size (N, )
        matched_mask: A tensor of size (N, )
        Return:
        matched_pred: A tensor of size (K, D), where K = sum(matched_mask)
        matched_gt: A tensor of size (K, D), where K = sum(matched_gt)
        """
        matched_pred = pred[matched_mask == 1, :]
        matched_gt = gt[gt_inds, :][matched_mask == 1, :]
        return matched_pred, matched_gt
    
    def get_para_loss(self, prediction, target)-> Dict[str, Tensor]:
        
        pred_para = prediction["matched"]["pred_pose_norm"]       
        para = target["matched"]["norm_pose"]

        para_loss = F.mse_loss(pred_para, para)
        loss = {"para": para_loss}

        return loss

    def get_noise_loss(self, prediction, target)-> Dict[str, Tensor]:
        
        pred_noise = prediction["matched"]["pred_noise"]       
        noise = prediction["matched"]["noise"]

        para_loss = F.mse_loss(pred_noise, noise)
        loss = {"noise": para_loss}

        return loss

    def get_hand_chamfer_loss(self, prediction, target) -> Dict[str, Tensor]:
        # chamfer loss between predict-hand point cloud and target-hand point cloud
        pred_hand_pc = prediction["hand"]["surface_points"]
        target_hand_pc = target["hand"]["surface_points"]
        chamfer_loss = chamfer_distance(pred_hand_pc, target_hand_pc, point_reduction="sum", batch_reduction="mean")[0]
        loss = {"hand_chamfer": chamfer_loss}
        return loss

    def get_cmap_loss(self, prediction, target) -> Dict[str, Tensor]:
        # chamfer loss between predict-hand point cloud and target-hand point cloud
        pred_hand_pc = prediction["hand"]["surface_points"]
        target_hand_pc = target["hand"]["surface_points"]
        pc = target["matched"]["obj_pc"]
        pred_cmap = contact_map_of_m_to_n(pc, pred_hand_pc)
        gt_cmap = contact_map_of_m_to_n(pc, target_hand_pc)
        cmap_loss = torch.nn.functional.mse_loss(pred_cmap, gt_cmap, reduction='mean')
        loss = {"cmap": cmap_loss}
        return loss

    def get_obj_penetration_loss(self, prediction, target) -> Dict[str, Tensor]:
        batch_size = prediction["hand"]["penetration_keypoints"].size(0)
        # signed squared distances from object_pc to hand, inside positive, outside negative
        distances = prediction["hand"]["penetration"]
        # loss_pen
        loss_pen = distances[distances > 0].sum() / batch_size
        loss = {"obj_penetration": loss_pen}
        return loss

    def get_self_penetration_loss(self, prediction, target) -> Dict[str, Tensor]:
        batch_size = prediction["hand"]["penetration_keypoints"].size(0)
        # loss_spen
        penetration_keypoints = prediction["hand"]["penetration_keypoints"]
        dis_spen = (penetration_keypoints.unsqueeze(1) - penetration_keypoints.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis_spen = torch.where(dis_spen < 1e-6, 1e6 * torch.ones_like(dis_spen), dis_spen)
        dis_spen = 0.02 - dis_spen
        dis_spen[dis_spen < 0] = 0
        loss_spen = dis_spen.sum() / batch_size
        loss = {"self_penetration": loss_spen}
        return loss
