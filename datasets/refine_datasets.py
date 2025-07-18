import glob
import json
import torch
import trimesh
import os.path as osp
import pytorch3d.transforms as T
from torch.functional import Tensor
from torch.utils.data import Dataset
from utils.rotation_utils import EulerConverter, RotNorm

class DgnBase(Dataset):
    ALL_CAT = [
        "apple", "banana", "binoculars", "bottle", "bowl", "cameras", "can", "cup",
        "cylinder_bottle", "donut", "eyeglasses", "flashlight", "fryingpan", "gamecontroller",
        "hammer", "headphones", "knife", "lightbulb", "lotion_pump", "mouse", "mug", "pen",
        "phone", "pincer", "power_drill", "scissors", "screwdriver", "squeezable", "stapler",
        "teapot", "toothbrush", "trigger_sprayer", "wineglass", "wrench"
    ]
    ALL_CAT_DCIT = {cat: idx for idx, cat in enumerate(ALL_CAT)}

    factor_minmax = torch.tensor(
        [[-0.22, 0.22], [-0.22, 0.22], [-0.22, 0.22],
         [-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14],
         [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
         [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
         [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
         [0.0, 0.785], [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
         [-1.047, 1.047], [0.0, 1.222], [-0.209, 0.209], [-0.524, 0.524], [-1.571, 0.0]]
    )

    factor_meastd = torch.tensor(
            [[-0.0084,  0.0834], [ 0.0160,  0.1095], [ 0.0133,  0.0761],
             [ 0.9996,  1.2288], [-0.6005,  0.5410], [-0.2261,  1.7977],
             [-0.1228,  0.1696], [ 0.5523,  0.3329], [ 0.4366,  0.2686],
             [ 0.5840,  0.2632], [-0.2177,  0.1015], [ 0.8147,  0.2867],
             [ 0.4558,  0.2614], [ 0.4287,  0.2877], [-0.2244,  0.0896],
             [ 0.8541,  0.2729], [ 0.4881,  0.2566], [ 0.4909,  0.2690],
             [ 0.1039,  0.0782], [-0.2459,  0.0279], [ 0.7202,  0.3355],
             [ 0.6243,  0.2199], [ 0.7668,  0.1824], [ 0.4145,  0.2530],
             [ 0.8502,  0.2004], [ 0.1375,  0.0322], [-0.0225,  0.2422],
             [-0.1882,  0.2151]])
    
    def __init__(self, rotation_type, norm_type) -> None:
        super().__init__()
        rot_transform = EulerConverter()
        rot_norm = RotNorm()
        if not hasattr(rot_transform, f'to_{rotation_type}'):
            raise NotImplementedError(f'Unsupported rotation type: {rotation_type}')
        else:
            self.rot_transform = getattr(rot_transform, f'to_{rotation_type}')
            self.rot_norm = getattr(rot_norm, f'norm_{rotation_type}')
        self.euler_to_axisangle = rot_transform.to_axisangle
        self.rotation_type = rotation_type
        self.norm_fuction = getattr(self, f"norm_by_{norm_type}")
        self.denorm_fuction = getattr(self, f"denorm_by_{norm_type}")
        self.norm_factor = getattr(self, f"factor_{norm_type[:6]}")

    @staticmethod
    def norm_by_minmax11(input: Tensor, minmax: Tensor) -> Tensor:
        """
        input: (N, C)
        minmax: (N, 2)
        """
        normed = 2 * (input - minmax[:, 0]) / (minmax[:, 1] - minmax[:, 0]) - 1 
        return normed

    @staticmethod
    def denorm_by_minmax11(input: Tensor, minmax: Tensor) -> Tensor:
        """
        input: (N, C)
        minmax: (N, 2)
        """
        denormed = 0.5 * (input + 1.0) * (minmax[:, 1] - minmax[:, 0]) + minmax[:, 0]
        return denormed
    
    @staticmethod
    def norm_by_minmax01(input: Tensor, minmax: Tensor) -> Tensor:
        """
        input: (N, C)
        minmax: (N, 2)
        """
        normed = (input - minmax[:, 0]) / (minmax[:, 1] - minmax[:, 0])
        return normed

    @staticmethod
    def denorm_by_minmax01(input: Tensor, minmax: Tensor) -> Tensor:
        """
        input: (N, C)
        minmax: (N, 2)
        """
        denormed  = input *(minmax[:, 1] - minmax[:, 0]) + minmax[:, 0]
        return denormed
    
    @staticmethod
    def norm_by_meastd11(input: Tensor, means_stds: Tensor) -> Tensor:
        """
        input: (N, C)
        means_stds: (N, 2)
        """
        means = means_stds[:, 0]
        stds = means_stds[:, 1]
        denormed = (input - means) / (2 * stds)
        return denormed

    @staticmethod
    def denorm_by_meastd11(input: Tensor, means_stds: Tensor) -> Tensor:
        """
        input: (N, C)
        means_stds: (N, 2)
        """
        means = means_stds[:, 0]
        stds = means_stds[:, 1]
        denormed = input * (2 * stds) + means 
        return denormed

    @staticmethod
    def collate_fn(batch):
        input_dict = {}
        for k in batch[0]:
            if k == "obj_pc":
                input_dict[k] = torch.stack([sample[k] for sample in batch])
            elif k == "assignments":
                if batch[0][k] is not None:
                    input_dict[k] = {
                        "per_query_gt_inds": torch.stack([torch.tensor(sample[k]["per_query_gt_inds"], dtype=torch.long) for sample in batch]),
                        "query_matched_mask": torch.stack([torch.tensor(sample[k]["query_matched_mask"], dtype=torch.long) for sample in batch]),
                    }
                else:
                    input_dict[k] = None
            else:
                input_dict[k] = [sample[k] for sample in batch]
        return input_dict

class RefineDataset(DgnBase):
    def __init__(
        self,
        data_root,
        pose_path,
        is_train=True,
        num_rotation_aug=1,
        num_equivalent_aug=1,
        sample_in_pose=True,
        norm_type="minmax11",
        guidance_type="simple",
        rotation_type="quaternion"
    ) -> None:
        super().__init__(rotation_type=rotation_type, norm_type=norm_type)

        self.data_root = data_root
        self.pose_path = pose_path
        self.is_train = is_train
        self.num_rotation_aug = num_rotation_aug
        self.rotation_type = rotation_type
        self.num_equivalent_aug = num_equivalent_aug
        self.sample_in_pose = sample_in_pose
        self.guidance_type = guidance_type
        self._process_data(sample_in_pose)

    def _process_data(self, sample_in_pose):
        # grasp
        with open(self.pose_path) as f:
            self.data = json.load(f)
        # obj
        data_dir = osp.join(self.data_root, "shape")
        oi_shape_dir = osp.join(data_dir, "oakink_shape_v2")
        meta_dir = osp.join(data_dir, "metaV2")
        self.real_meta = json.load(open(osp.join(meta_dir, "object_id.json"), "r"))
        self.virtual_meta = json.load(open(osp.join(meta_dir, "virtual_object_id.json"), "r"))

    def __getitem__(self, index):

        data = self.data[index]
        cate_id = data["cate_id"]
        obj_id = data["obj_id"]
        guidance = data["guidance"]
        
        obj_pc = self._get_obj_pc(obj_id)

        def _norm(pose):
            hand_translation = pose[:3]
            hand_axis_angle = pose[3:6]
            hand_qpos = pose[6:]

            # norm
            norm_qpos = self.norm_fuction(hand_qpos, self.norm_factor[6:])
            hand_euler = torch.flip(
                T.matrix_to_euler_angles(T.axis_angle_to_matrix(hand_axis_angle), "ZYX"),
                dims=[-1]
            )
            norm_rotation = self.rot_transform(hand_euler)
            if self.rotation_type == "euler":
                norm_rotation = self.norm_fuction(norm_rotation, self.norm_factor[3:6])
            norm_translation = self.norm_fuction(hand_translation, self.norm_factor[:3])
            norm_pose = torch.cat([norm_translation, norm_qpos, norm_rotation], dim=-1)
            return norm_pose

        coarse_pose = torch.tensor(data["predictions"])
        coarse_norm_pose = _norm(coarse_pose)
        fine_pose = torch.tensor(data["targets"])
        fine_norm_pose = _norm(fine_pose)
        
        sample = {
            "obj_id": obj_id,
            "cate_id": data["cate_id"],
            "guidance": data["guidance"],
            "obj_pc": obj_pc,  # shape: (N, 3)
            "intend_id": data["action_id"],
            "coarse_norm_pose": coarse_norm_pose,  
            "coarse_pose": coarse_pose,  
            "norm_pose": fine_norm_pose,  
            "hand_model_pose": fine_pose,  
            "rotation_type": self.rotation_type,
        }

        return sample

    def _get_obj_pc(self, oid, use_downsample=True, key="align"):
        data_dir = osp.join(self.data_root, "shape")
        obj_suffix_path = "align_ds" if use_downsample else "align"
        if oid in self.real_meta:
            obj_name = self.real_meta[oid]["name"]
            obj_path = osp.join(data_dir, "OakInkObjectsV2")
        else:
            obj_name = self.virtual_meta[oid]["name"]
            obj_path = osp.join(data_dir, "OakInkVirtualObjectsV2")
        obj_mesh_path = list(
            glob.glob(osp.join(obj_path, obj_name, obj_suffix_path, "*.obj")) +
            glob.glob(osp.join(obj_path, obj_name, obj_suffix_path, "*.ply")))
        if len(obj_mesh_path) > 1:
            obj_mesh_path = [p for p in obj_mesh_path if key in osp.split(p)[1]]
        assert len(obj_mesh_path) == 1
        obj_path = obj_mesh_path[0]   
        obj_trimesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
        bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
        obj_trimesh.vertices = obj_trimesh.vertices - bbox_center
        obj_pc = torch.tensor(obj_trimesh.sample(4096)).float()
        return obj_pc  
    
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
    
    def __len__(self,):
        return len(self.data)



    