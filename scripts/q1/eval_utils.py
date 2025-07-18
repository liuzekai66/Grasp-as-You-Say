"""
Last modified date: 2022.09.09
Author: mzhmxzh
Description: cal q1 for datasetv3f
"""

import argparse
import json
import os
import os.path as osp
import sys
from statistics import mean
from typing import Dict

import glob
import csdf
import trimesh
import numpy as np
import pytorch3d.transforms as T
import scipy.spatial
import torch
from csdf import compute_sdf, index_vertices_by_faces
from torch.functional import Tensor
from tqdm import tqdm, trange


sys.path.append("./")
from utils.config_utils import EasyConfig
from utils.shadowhand import ShallowHandModel


translation_names = ["WRJTx", "WRJTy", "WRJTz"]
rot_names = ["WRJRx", "WRJRy", "WRJRz"]
joint_names = [
    "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
    "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
    "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
    "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
    "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"
]


class KaolinModel:

    def __init__(self, data_root_path, batch_size_each, device="cuda"):

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path

        self.data_dir = os.path.join(data_root_path, "shape")
        self.oi_shape_dir = os.path.join(self.data_dir, "oakink_shape_v2")
        self.meta_dir = os.path.join(self.data_dir, "metaV2")

        
        self.real_meta = json.load(open(os.path.join(self.meta_dir, "object_id.json"), "r"))
        self.virtual_meta = json.load(open(os.path.join(self.meta_dir, "virtual_object_id.json"), "r"))


    def get_obj_path(self, oid, use_downsample=True, key="align"):
        obj_suffix_path = "align_ds" if use_downsample else "align"
        if oid in self.real_meta:
            obj_name = self.real_meta[oid]["name"]
            obj_path = os.path.join(self.data_dir, "OakInkObjectsV2")
        else:
            obj_name = self.virtual_meta[oid]["name"]
            obj_path = os.path.join(self.data_dir, "OakInkVirtualObjectsV2")
        obj_mesh_path = list(
            glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj")) +
            glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply")))
        if len(obj_mesh_path) > 1:
            obj_mesh_path = [p for p in obj_mesh_path if key in os.path.split(p)[1]]
        

        assert len(obj_mesh_path) == 1, (len(obj_mesh_path), oid, obj_name)
        return obj_mesh_path[0]

    def initialize(self, object_id):
        obj_path = self.get_obj_path(object_id)
        obj_trimesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
        bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
        obj_trimesh.vertices = obj_trimesh.vertices - bbox_center
        self.object_face_verts_list = [index_vertices_by_faces(
                                            torch.tensor(obj_trimesh.vertices).to(self.device, torch.float32), 
                                            torch.tensor(obj_trimesh.faces).to(self.device, torch.long))
                                        ]
        self.surface_points_tensor = torch.tensor(obj_trimesh.sample(4096)).to(dtype=torch.float32, device=self.device).unsqueeze(0)

    def cal_distance(self, x, with_closest_points=False):
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance = []
        normals = []
        closest_points = []
        for i in range(x.shape[0]):
            face_verts = self.object_face_verts_list[i]
            dis, normal, dis_signs, _, _ = csdf.compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis+1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)
        if with_closest_points:
            closest_points = torch.stack(closest_points).reshape(-1, n_points, 3)
            return distance, normals, closest_points
        return distance, normals


def cal_q1(
    cfg: Dict,
    hand_model,
    object_model,
    object_code: str,
    hand_pose: Tensor,
    device
):
    """
    calculate Q1 metric for one object and one grasp at a time

    Params:
        cfg: a configuration dictionary for Q1 metric
        hand_model: a HandModel instance
        object_model: an ObjectModel instance
        objecet_code: an object code string
        scale: a float number for object scale
        hand_pose: a tensor of size (28, )  (translation(3) + axisangle(3) + joints(22))
    Return:
        A float number representing the Q1 metric of given object and hand pose.
    """
    # load data
    object_model.initialize(object_code)
    object_model.batch_size_each = 1
    # cal hand
    hand_pose = hand_pose.unsqueeze(0)
    global_translation = hand_pose[:, 0:3]
    global_rotation = T.axis_angle_to_matrix(hand_pose[:, 3:6])
    current_status = hand_model.chain.forward_kinematics(hand_pose[:, 6:])
    # cal contact points and contact normals
    contact_points_object = []
    contact_normals = []
    for link_name in hand_model.mesh:
        if len(hand_model.mesh[link_name]["surface_points"]) == 0:
            continue
        surface_points = current_status[link_name].transform_points(hand_model.mesh[link_name]["surface_points"])
        surface_points = surface_points @ global_rotation.transpose(1, 2) + global_translation.unsqueeze(1)
        distances, normals, closest_points = object_model.cal_distance(surface_points, with_closest_points=True)
        if cfg["nms"]:
            nearest_point_index = distances.argmax()
            if -distances[0, nearest_point_index] < cfg["thres_contact"]:
                contact_points_object.append(closest_points[0, nearest_point_index])
                contact_normals.append(normals[0, nearest_point_index])
        else:
            contact_idx = (-distances < cfg["thres_contact"]).nonzero().reshape(-1)
            if len(contact_idx) != 0:
                for idx in contact_idx:
                    contact_points_object.append(closest_points[0, idx])
                    contact_normals.append(normals[0, idx])
    if len(contact_points_object) == 0:
        contact_points_object.append(torch.tensor([0, 0, 0], dtype=torch.float, device=device))
        contact_normals.append(torch.tensor([1, 0, 0], dtype=torch.float, device=device))

    contact_points_object = torch.stack(contact_points_object).cpu().numpy()
    contact_normals = torch.stack(contact_normals).cpu().numpy()

    n_contact = len(contact_points_object)

    if np.isnan(contact_points_object).any() or np.isnan(contact_normals).any():
        return 0

    # cal contact forces
    u1 = np.stack([
        -contact_normals[:, 1],
        contact_normals[:, 0],
        np.zeros([n_contact], dtype=np.float32),
    ], axis=1)
    u2 = np.stack([
        np.ones([n_contact], dtype=np.float32),
        np.zeros([n_contact], dtype=np.float32),
        np.zeros([n_contact], dtype=np.float32),
    ], axis=1)
    u = np.where(np.linalg.norm(u1, axis=1, keepdims=True) > 1e-8, u1, u2)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(u, contact_normals)
    theta = np.linspace(0, 2 * np.pi, cfg["m"], endpoint=False).reshape(cfg["m"], 1, 1)
    contact_forces = (contact_normals + cfg["mu"] * (np.cos(theta) * u + np.sin(theta) * v)).reshape(-1, 3)

    # cal wrench space and q1
    origin = np.array([0, 0, 0], dtype=np.float32)
    wrenches = np.concatenate([
        np.concatenate([
            contact_forces,
            cfg["lambda_torque"] * np.cross(np.tile(contact_points_object - origin, (cfg["m"], 1)), contact_forces)
        ], axis=1),
        np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float32)
    ], axis=0)
    try:
        wrench_space = scipy.spatial.ConvexHull(wrenches)
    except scipy.spatial._qhull.QhullError:
        return 0
    q1 = np.array([1], dtype=np.float32)
    for equation in wrench_space.equations:
        q1 = np.minimum(q1, np.abs(equation[6]) / np.linalg.norm(equation[:6]))

    return q1.item()


def cal_pen(hand_model, object_model, object_code, hand_pose, device):
    # load data
    object_model.initialize(object_code)
    object_model.batch_size_each = 1
    # cal pen
    object_surface_points = object_model.surface_points_tensor
    hand_pose = hand_pose.unsqueeze(0)
    global_translation = hand_pose[:, 0:3]
    global_rotation = T.axis_angle_to_matrix(hand_pose[:, 3:6])
    current_status = hand_model.chain.forward_kinematics(hand_pose[:, 6:])
    distances = []
    x = (object_surface_points - global_translation.unsqueeze(1)) @ global_rotation
    for link_name in hand_model.mesh:
        if link_name in ["robot0:forearm", "robot0:wrist_child", "robot0:ffknuckle_child", "robot0:mfknuckle_child", "robot0:rfknuckle_child", "robot0:lfknuckle_child", "robot0:thbase_child", "robot0:thhub_child"]:
            continue
        matrix = current_status[link_name].get_matrix()
        x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
        x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
        if "geom_param" not in hand_model.mesh[link_name]:
            face_verts = hand_model.mesh[link_name]["face_verts"]
            dis_local, _, dis_signs, _, _ = compute_sdf(x_local, face_verts)
            dis_local = torch.sqrt(dis_local + 1e-8)
            dis_local = dis_local * (-dis_signs)
        else:
            height = hand_model.mesh[link_name]["geom_param"][1] * 2
            radius = hand_model.mesh[link_name]["geom_param"][0]
            nearest_point = x_local.detach().clone()
            nearest_point[:, :2] = 0
            nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], 0, height)
            dis_local = radius - (x_local - nearest_point).norm(dim=1)
        distances.append(dis_local.reshape(x.shape[0], x.shape[1]))
    distances = torch.max(torch.stack(distances, dim=0), dim=0)[0]

    return max(distances.max().item(), 0)


def eval_result(q1_cfg, result, hand_model, object_model, device):
    per_object_metric = {f'{r["obj_id"]}_{r["guidance"]}': [] for r in result}
    overall_metric = {"q1": [], "pen": [], "valid_q1": []}

    pbar = tqdm(total=len(result), desc="Evaluate", ncols=120)
    for res in result:
        object_code = res["obj_id"]
        action_id = res["guidance"]
        hand_pose = torch.tensor(res["predictions"], device=device)
        if hand_pose.dim() == 3:
            hand_pose = hand_pose.squeeze(1)
        elif hand_pose.dim() == 1:
            hand_pose = hand_pose.unsqueeze(0)

        for i in trange(hand_pose.size(0), desc=object_code, ncols=120, leave=False):
            q1 = cal_q1(
                q1_cfg,
                hand_model,
                object_model,
                object_code,
                hand_pose[i],
                device
            )
            pen = cal_pen(
                hand_model,
                object_model,
                object_code,
                hand_pose[i],
                device
            )
            valid = (pen < q1_cfg["thres_pen"])
            valid_q1 = q1 if valid else 0

            overall_metric["q1"].append(q1)
            overall_metric["pen"].append(pen)
            overall_metric["valid_q1"].append(valid_q1)
            per_object_metric[f'{object_code}_{action_id}'].append({
                "q1": q1,
                "pen": pen,
                "valid_q1": valid_q1,
            })

        # update pbar
        _mean = {k: round(mean(v), 2) for k, v in overall_metric.items()}
        pbar.set_postfix(_mean)
        pbar.update()

    # save metrics
    object_detail_metric = {}
    for obj_code_with_scale, metrics in per_object_metric.items():
        obj_mean = {
            "q1": mean([x["q1"] for x in metrics]),
            "pen": mean([x["pen"] for x in metrics]),
            "valid_q1": mean([x["valid_q1"] for x in metrics]),
        }
        obj_max_pen = max([x["pen"] for x in metrics])
        object_detail_metric[obj_code_with_scale] = {"max_pen": obj_max_pen, "mean": obj_mean, "detail": metrics}
    # mean of obj_mean
    mean_obj_metric = {
        "q1": mean([x["mean"]["q1"] for x in object_detail_metric.values()]),
        "pen": mean([x["mean"]["pen"] for x in object_detail_metric.values()]),
        "valid_q1": mean([x["mean"]["valid_q1"] for x in object_detail_metric.values()]),
    }
    # mean of overall grasps
    mean_overall_metric = {
        "q1": mean(overall_metric["q1"]),
        "pen": mean(overall_metric["pen"]),
        "valid_q1": mean(overall_metric["valid_q1"]),
    }
    # save result dict
    result_dict = {
        "overall_max_pen": max(overall_metric["pen"]),
        "mean_object": mean_obj_metric,
        "mean_overall": mean_overall_metric,
        "object_details": object_detail_metric,
    }
    return result_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_path", type=str, required=True, help="path for results.json saved by test.py")
    parser.add_argument("-g", "--gpu", type=int)
    parser.add_argument("-s", "--start", type=int, help="Only results[start: end] will be evaluated")
    parser.add_argument("-e", "--end", type=int, help="Only results[start: end] will be evaluated")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ['OAKINK_DIR'] = "data/oakink"

    with open(args.result_path) as rf:
        results = json.load(rf)

    # new_results = []
    # for k in results:
    #     new_results.append(results[k])
    # with open("results/result_list.json",'w') as file:
    #     json.dump(new_results, file, indent=4)

    if args.start is not None and args.end is not None:
        results = results[args.start: args.end]
    save_path = osp.join(osp.dirname(osp.abspath(args.result_path)), f"metrics_{args.start}_{args.end}.json")
    
    cfg = EasyConfig()
    cfg.load("config/test_base.yaml")
    hand_model = ShallowHandModel(device="cuda")
    object_model = KaolinModel(
        os.environ['OAKINK_DIR'],
        batch_size_each=1,
        device="cuda",
    )
    
    metric_dict = eval_result(cfg.q1, results, hand_model, object_model, device="cuda")
    with open(save_path, "w") as wf:
        json.dump(metric_dict, wf, indent=4)
