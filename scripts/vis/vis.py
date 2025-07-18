import os
import sys
import glob
import json
import tqdm
import torch
import trimesh
import plotly.graph_objects as go

sys.path.append("./")
from model.utils.hand_model import HandModel

class Hand_Config:
    def __init__(self):
        self.mjcf_path = "./data/mjcf/shadow_hand.xml"
        self.mesh_path = "./data/mjcf/meshes"
        self.n_surface_points = 1024
        self.contact_points_path = "./data/mjcf/contact_points.json"
        self.penetration_points_path = "./data/mjcf/penetration_points.json"
        self.fingertip_points_path = "./data/mjcf/fingertip.json"


class Object_mesh:
    def __init__(self) -> None:
        data_root_path = "/home/yilin/2024/TOGG/data"
        self.data_dir = os.path.join(data_root_path, "shape")
        oi_shape_dir = os.path.join(self.data_dir, "oakink_shape_v2")
        meta_dir = os.path.join(self.data_dir, "metaV2")
        self.obj_suffix_path = "align"
        self.real_meta = json.load(open(os.path.join(meta_dir, "object_id.json"), "r"))
        self.virtual_meta = json.load(open(os.path.join(meta_dir, "virtual_object_id.json"), "r"))

    def get_obj_mesh(self, oid, use_downsample=False, key="align"):
        if oid in self.real_meta:
            obj_name = self.real_meta[oid]["name"]
            atrri = self.real_meta[oid]["attr"]
            obj_path = os.path.join(self.data_dir, "OakInkObjectsV2")
        else:
            obj_name = self.virtual_meta[oid]["name"]
            atrri = self.virtual_meta[oid]["attr"]
            obj_path = os.path.join(self.data_dir, "OakInkVirtualObjectsV2")
        # if "pump" not in obj_name:
        #     return "0", 0
        obj_mesh_path = list(
            glob.glob(os.path.join(obj_path, obj_name, self.obj_suffix_path, "*.obj")) +
            glob.glob(os.path.join(obj_path, obj_name, self.obj_suffix_path, "*.ply")))
        if len(obj_mesh_path) > 1:
            obj_mesh_path = [p for p in obj_mesh_path if key in os.path.split(p)[1]]
        assert len(obj_mesh_path) == 1
        obj_path = obj_mesh_path[0]   
        obj_trimesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
        bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
        obj_trimesh.vertices = obj_trimesh.vertices - bbox_center
        obj_mesh = {"vertices": obj_trimesh.vertices, "faces": obj_trimesh.faces}
        return obj_name, obj_mesh  

if __name__ == "__main__":
    hand_config = Hand_Config()
    hand_model = HandModel(hand_config, "cpu")

    with open("data/dexgys/train.json") as f:
        data_list = json.load(f)

    save_root = os.path.join("./Experiments/vis_mesh", "pred2")
    os.makedirs(save_root, exist_ok=True)

    object_class = Object_mesh()
    vis_dict = {}
    for i in tqdm.tqdm(range(0, len(data_list), 1)):
        data = data_list[i]
        guidence = data['guidance']
        obj_id = data['cate_id']
        intend_id = data['action_id']
        intend_fun = {
                "0001": "use",
                "0002": "hold",
                "0003": "lift",
                "0004": "hand_over"
            }

        intend = intend_fun[intend_id]
        only_code = f"{obj_id}_{intend}"

        if only_code in vis_dict:
            continue
        if only_code in vis_dict:
            vis_dict[only_code] += 1
        else:
            vis_dict[only_code] = 1

        obj_name, object_mesh = object_class.get_obj_mesh(data["obj_id"])

        if "dex_grasp" in data:
            predictions = torch.tensor(data["dex_grasp"]).unsqueeze(0)
        else:
            predictions = torch.tensor(data["predictions"]).unsqueeze(0)
        
        hand = hand_model(predictions, with_meshes=True)
        hand['vertices'] = hand['vertices'].detach().cpu()
        hand['faces'] = hand['faces'].detach().cpu()

        objmesh = trimesh.Trimesh(vertices=object_mesh['vertices'], faces=object_mesh['faces'])
        handmesh = trimesh.Trimesh(vertices=hand['vertices'][0], faces=hand['faces'])

        combined_mesh = objmesh
        for j in range(predictions.shape[0]):
            handmesh = trimesh.Trimesh(vertices=hand['vertices'][j], faces=hand['faces'])
            combined_mesh += handmesh
            break

        combined_mesh.export(os.path.join(save_root, f"{only_code}_{str(vis_dict[only_code])}.obj"))





