import os
import sys
import json
import glob
import tqdm
import torch
import trimesh
import argparse
import numpy as np
from torch.functional import Tensor
from torch.utils.data import Dataset, DataLoader

from point_e.evals.fid_is import compute_statistics
sys.path.append("./")
from model.utils.hand_model import HandModel
from scripts.fid.feature_extractor import get_model


class Hand_Config:
    def __init__(self):
        self.mjcf_path = "./data/mjcf/shadow_hand.xml"
        self.mesh_path = "./data/mjcf/meshes"
        self.n_surface_points = 1024
        self.contact_points_path = "./data/mjcf/contact_points.json"
        self.penetration_points_path = "./data/mjcf/penetration_points.json"
        self.fingertip_points_path = "./data/mjcf/fingertip.json"

def get_obj_points(oid, use_downsample=True, key="align"):

    data_root_path = "/home/yilin/2024/TOGG/data"
    data_dir = os.path.join(data_root_path, "shape")
    oi_shape_dir = os.path.join(data_dir, "oakink_shape_v2")
    meta_dir = os.path.join(data_dir, "metaV2")
    obj_suffix_path = "align_ds" if use_downsample else "align"
    real_meta = json.load(open(os.path.join(meta_dir, "object_id.json"), "r"))
    virtual_meta = json.load(open(os.path.join(meta_dir, "virtual_object_id.json"), "r"))

    if oid in real_meta:
        obj_name = real_meta[oid]["name"]
        obj_path = os.path.join(data_dir, "OakInkObjectsV2")
    else:
        obj_name = virtual_meta[oid]["name"]
        obj_path = os.path.join(data_dir, "OakInkVirtualObjectsV2")

    obj_mesh_path = list(
        glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj")) +
        glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply")))
    if len(obj_mesh_path) > 1:
        obj_mesh_path = [p for p in obj_mesh_path if key in os.path.split(p)[1]]
    assert len(obj_mesh_path) == 1
    obj_path = obj_mesh_path[0]   
    obj_trimesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
    bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
    obj_trimesh.vertices = obj_trimesh.vertices - bbox_center
    points = trimesh.sample.sample_surface(obj_trimesh, 1024)
    points = torch.tensor(points[0], dtype=torch.float32)
    return points

class DictDataset(Dataset):

    # "binoculars", 
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = {}
        if "predictions" in self.data[index]:
            data["predictions"] = torch.tensor(self.data[index]["predictions"]).squeeze()
            while data["predictions"].shape[0] != 28:
                data["predictions"] = data["predictions"][0]
        elif "dex_grasp" in self.data[index]:
            data["dex_grasp"] = torch.tensor(self.data[index]["dex_grasp"]).squeeze()
            # if data["dex_grasp"].shape[0] != 28:
            #     print(data)
            #     raise Exception("1")
        else:
            raise Exception("1")


        data["obj_pc"] = get_obj_points(self.data[index]["obj_id"])
        return data
        
    @staticmethod
    def collate_fn(batch):
        input_dict = {}
        for k in batch[0]:
            if isinstance(batch[0][k], Tensor):
                try:
                    input_dict[k] = torch.stack([sample[k] for sample in batch])
                except:
                    input_dict[k] = [sample[k] for sample in batch]
            else:
                input_dict[k] = [sample[k] for sample in batch]
        return input_dict

def normalize_point_clouds(pc: torch.Tensor) -> torch.Tensor:
    centroids = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroids
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
    pc = pc / m
    return pc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument('-r', '--json_path', type=str, default=1,
                        help='an integer for the accumulator')
    args = parser.parse_args()
    device = "cuda"
    batch_size = 60
    gt_path = "data/dexgys/test.json"

    hand_config = Hand_Config()
    dexhandmodel = HandModel(cfg=hand_config ,device=device)
    feature_extractor = get_model()

    with open(args.json_path) as f:
        pred_data = json.load(f)
    pred_dataset = DictDataset(pred_data)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, collate_fn=pred_dataset.collate_fn, num_workers=8, shuffle=True)

    with open(gt_path) as f:
        gt_data = json.load(f)
    gt_dataset = DictDataset(gt_data)
    gt_loader = DataLoader(gt_dataset, batch_size=batch_size, collate_fn=gt_dataset.collate_fn, num_workers=8, shuffle=True)


    pred_pc = []
    pred_pc_wobj = []
    gt_pc = []
    gt_pc_wobj = []
    fids = []
    features_pred = []

    for i, batch in tqdm.tqdm(enumerate(pred_loader), total=len(pred_loader)):
        obj_pcs = batch["obj_pc"].to(device)
        # dex
        dexhand_pc = dexhandmodel(batch["predictions"].to(device), with_surface_points=True)["surface_points"]
        
        # input_pc = dexhand_pc.transpose(-1,-2)
        input_pc = normalize_point_clouds(torch.cat([obj_pcs, dexhand_pc], dim=1)).transpose(-1,-2)
        _, _, features = feature_extractor(input_pc, features=True)
        features_pred.append(features.cpu().detach())


    features_gt = []
    for i, batch in tqdm.tqdm(enumerate(gt_loader), total=len(gt_loader)):
        obj_pcs = batch["obj_pc"].to(device)
        # dex
        dexhand_pc = dexhandmodel(batch["dex_grasp"].to(device), with_surface_points=True)["surface_points"]
        
        # input_pc = dexhand_pc.transpose(-1,-2)
        input_pc = normalize_point_clouds(torch.cat([obj_pcs, dexhand_pc], dim=1)).transpose(-1,-2)
        _, _, features = feature_extractor(input_pc, features=True)
        features_gt.append(features.cpu().detach())

    features_pred = torch.cat(features_pred, dim=0).numpy()
    stats_p = compute_statistics(features_pred)

    features_gt = torch.cat(features_gt, dim=0).numpy()
    stats_gt = compute_statistics(features_gt)

    fid = stats_p.frechet_distance(stats_gt)
    print(fid)
