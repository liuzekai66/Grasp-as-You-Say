import argparse
import json
import os
import os.path as osp
from statistics import mean
from typing import Dict, List, Tuple

import pytorch3d.transforms as T
import torch
from torch.functional import Tensor


class DiversityEvalator:
    joint_limits = torch.tensor([
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [0.0, 0.785], [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-1.047, 1.047], [0.0, 1.222], [-0.209, 0.209], [-0.524, 0.524], [-1.571, 0.0]
    ])  # (22, 2)

    def __init__(self, results) -> None:
        self.results = results
        self.metric = {"details": {}}

    def calc_diversity(self):
        overall_preds = {"translations": [], "joint_angles": [], 'rotation': []}
        per_obj_diversity = {
            "var_translation": [], "var_joint_angle": [], "var_rotation": [],
            "std_translation": [], "std_joint_angle": [], "std_rotation": [],
            "real_var_joint_angle": [], "real_var_rotation": [],
            "real_std_joint_angle": [], "real_std_rotation": [],
        }
        for res in self.results:
            obj_code_with_scale = res["obj_id"]
            # scale = res["scale"]
            # obj_code_with_scale = f"{obj_code}_{scale}"
            predictions = torch.tensor(res["predictions"], dtype=torch.float).squeeze(1)
            # print(predictions.shape)

            translations = predictions[:, :3]
            translations = translations * 100.0

            rotation = predictions[:, 3:6]
            euler_angle = T.matrix_to_euler_angles(T.axis_angle_to_matrix(rotation), "XYZ")
            rotation = torch.rad2deg(euler_angle)

            joint_angles = predictions[:, -22:]
            joint_angles = joint_angles.clamp(min=self.joint_limits[:, 0], max=self.joint_limits[:, 1])
            joint_angles = torch.rad2deg(joint_angles)

            # real var and std
            real_var_joints, real_std_joints = self._calc_rotation_var_std(joint_angles)
            real_var_rotation, real_std_rotation = self._calc_rotation_var_std(rotation)

            mean_var_trans = torch.var(translations, dim=0, unbiased=True).mean().item()
            mean_var_joints = torch.var(joint_angles, dim=0, unbiased=True).mean().item()
            mean_var_rotation = torch.var(rotation, dim=0, unbiased=True).mean().item()
            mean_std_trans = torch.std(translations, dim=0, unbiased=True).mean().item()
            mean_std_joints = torch.std(joint_angles, dim=0, unbiased=True).mean().item()
            mean_std_rotation = torch.std(rotation, dim=0, unbiased=True).mean().item()
            mean_real_var_joints = real_var_joints.mean().item()
            mean_real_std_joints = real_std_joints.mean().item()
            mean_real_var_rotation = real_var_rotation.mean().item()
            mean_real_std_rotation = real_std_rotation.mean().item()

            # update per object metric detail
            self.metric["details"][obj_code_with_scale] = {"mean": {}}
            obj_mean_metric: Dict = self.metric["details"][obj_code_with_scale]["mean"]
            obj_mean_metric.update({
                "var_translation": mean_var_trans,
                "var_joint_angle": mean_var_joints,
                "var_rotation": mean_var_rotation,
            })
            self.metric["details"][obj_code_with_scale]["mean"] = obj_mean_metric
            # record for mean-object metrics
            per_obj_diversity["var_translation"].append(mean_var_trans)
            per_obj_diversity["var_joint_angle"].append(mean_var_joints)
            per_obj_diversity["var_rotation"].append(mean_var_rotation)
            per_obj_diversity["std_translation"].append(mean_std_trans)
            per_obj_diversity["std_joint_angle"].append(mean_std_joints)
            per_obj_diversity["std_rotation"].append(mean_std_rotation)
            per_obj_diversity["real_var_joint_angle"].append(mean_real_var_joints)
            per_obj_diversity["real_var_rotation"].append(mean_real_var_rotation)
            per_obj_diversity["real_std_joint_angle"].append(mean_real_std_joints)
            per_obj_diversity["real_std_rotation"].append(mean_real_std_rotation)
            # record for overall metrics
            overall_preds["translations"].append(translations)
            overall_preds["joint_angles"].append(joint_angles)
            overall_preds["rotation"].append(rotation)

        overall_diversity = {
            "mean_var_trans": torch.var(torch.cat(overall_preds["translations"]), dim=0, unbiased=True).mean().item(),
            "mean_var_joints": torch.var(torch.cat(overall_preds["joint_angles"]), dim=0, unbiased=True).mean().item(),
            "mean_var_rotation": torch.var(torch.cat(overall_preds["rotation"]), dim=0, unbiased=True).mean().item(),
            "mean_std_trans": torch.std(torch.cat(overall_preds["translations"]), dim=0, unbiased=True).mean().item(),
            "mean_std_joints": torch.std(torch.cat(overall_preds["joint_angles"]), dim=0, unbiased=True).mean().item(),
            "mean_std_rotation": torch.std(torch.cat(overall_preds["rotation"]), dim=0, unbiased=True).mean().item(),
            "mean_real_var_joint_angle": self._calc_rotation_var_std(torch.cat(overall_preds["joint_angles"]))[0].mean().item(),
            "mean_real_var_rotation": self._calc_rotation_var_std(torch.cat(overall_preds["rotation"]))[0].mean().item(),
            "mean_real_std_joint_angle": self._calc_rotation_var_std(torch.cat(overall_preds["joint_angles"]))[1].mean().item(),
            "mean_real_std_rotation": self._calc_rotation_var_std(torch.cat(overall_preds["rotation"]))[1].mean().item(),
        }
        mean_object_diversity = {
            "mean_var_trans": mean(per_obj_diversity["var_translation"]),
            "mean_var_joints": mean(per_obj_diversity["var_joint_angle"]),
            "mean_var_rotation": mean(per_obj_diversity["var_rotation"]),
            "mean_std_trans": mean(per_obj_diversity["std_translation"]),
            "mean_std_joints": mean(per_obj_diversity["std_joint_angle"]),
            "mean_std_rotation": mean(per_obj_diversity["std_rotation"]),
            "mean_real_var_joint_angle": mean(per_obj_diversity["real_var_joint_angle"]),
            "mean_real_var_rotation": mean(per_obj_diversity["real_var_rotation"]),
            "mean_real_std_joint_angle": mean(per_obj_diversity["real_std_joint_angle"]),
            "mean_real_std_rotation": mean(per_obj_diversity["real_std_rotation"]),
        }
        final_metric = {
            "diversity": {
                "overall": overall_diversity,
                "mean_object": mean_object_diversity,
            }
        }
        final_metric.update(self.metric)

        return final_metric

    def _calc_rotation_var_std(self, samples: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Calculate variance and stand deviation (unbiased).
        Optimized (X - mean) for rotations.

        Args:
            samples (Tensor): a Tensor of N rotation samples (N, C)

        Returns:
            Tuple[Tensor, Tensor]: variance and std of the samples (C, )
        """
        sample_mean = torch.mean(samples, 0, keepdim=True)  # (1, C)
        difference = (samples - sample_mean).abs()
        difference = torch.where(difference <= 180.0, difference, 360.0 - difference)
        variance = difference.pow(2).sum(0) / (samples.size(0) - 1)
        std = variance.sqrt()
        return variance, std

    def _calc_entropy(self, samples: Tensor) -> float:
        """
        1. divide the motion range of each joint into 100 bins
        2. calculate the distribution for each joint
        3. calculate the entropy for each joint
        4. average over joints

        Params:
            samples: A tensor of shape (N, 22)
        Returns:
            mean entropy over all joints
        """
        result = torch.empty(22, dtype=torch.float)
        for i in range(22):
            _samples = samples[:, i]
            prob = torch.histogram(_samples, 100, range=(self.joint_limits[i].min().item(), self.joint_limits[i].max().item()))[0]
            prob /= prob.sum()
            distribution = torch.distributions.Categorical(probs=prob)
            result[i] = distribution.entropy()
        return result.mean().item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_path", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Usage: python ./calc_diversity.py -r results.json
    """
    args = parse_args()
    save_path = osp.join(osp.dirname(args.result_path), "diversity_with_std.json")
    if osp.isfile(save_path):
        input(f"{save_path} already exists.\nPress Enter to delete it or Ctrl-C to quit.")
        os.remove(save_path)
    with open(args.result_path) as rf:
        data = json.load(rf)

    diversity = DiversityEvalator(data)
    final_result = diversity.calc_diversity()

    with open(save_path, 'w') as f:
        json.dump(final_result, f, indent=4)
