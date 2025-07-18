from typing import Optional, Tuple

from torch.utils.data import Dataset
from .task_dex_datasets import TaskDataset_Pose
from .refine_datasets import RefineDataset


def build_datasets(data_cfg) -> Tuple[Dataset, Dataset, Optional[Dataset]]:

    if data_cfg.name.lower() == "task_pose":
        if not hasattr(data_cfg, "test"):
            train_set = TaskDataset_Pose(
                        data_root=data_cfg.train.data_root,
                        pose_path=data_cfg.train.pose_path,
                        rotation_type=data_cfg.train.rotation_type,
                        sample_in_pose = data_cfg.train.sample_in_pose,
                        norm_type=data_cfg.train.norm_type,
                        guidance_type=data_cfg.train.guidance_type,
                        is_train=True,
                )
            val_set = TaskDataset_Pose(
                        data_root=data_cfg.val.data_root,
                        pose_path=data_cfg.val.pose_path,
                        rotation_type=data_cfg.val.rotation_type,
                        sample_in_pose = data_cfg.val.sample_in_pose,
                        norm_type=data_cfg.val.norm_type,
                        guidance_type=data_cfg.val.guidance_type,
                        is_train=False,
            )
            test_set = None
        elif hasattr(data_cfg, "test"):
            train_set = None
            val_set = None
            test_set = TaskDataset_Pose(
                        data_root=data_cfg.test.data_root,
                        pose_path=data_cfg.test.pose_path,
                        rotation_type=data_cfg.test.rotation_type,
                        sample_in_pose = data_cfg.test.sample_in_pose,
                        norm_type=data_cfg.test.norm_type,
                        guidance_type=data_cfg.test.guidance_type,
                        is_train=False,
            )

        else:
            raise Exception("1")
        return train_set, val_set, test_set
    elif data_cfg.name.lower() == "refinement":
        if not hasattr(data_cfg, "test"):
            train_set = RefineDataset(
                        data_root=data_cfg.train.data_root,
                        pose_path=data_cfg.train.pose_path,
                        rotation_type=data_cfg.train.rotation_type,
                        sample_in_pose = data_cfg.train.sample_in_pose,
                        norm_type=data_cfg.train.norm_type,
                        guidance_type=data_cfg.train.guidance_type,
                        is_train=True,
                )
            val_set = RefineDataset(
                        data_root=data_cfg.val.data_root,
                        pose_path=data_cfg.val.pose_path,
                        rotation_type=data_cfg.val.rotation_type,
                        sample_in_pose = data_cfg.val.sample_in_pose,
                        norm_type=data_cfg.val.norm_type,
                        guidance_type=data_cfg.val.guidance_type,
                        is_train=False,
            )
            test_set = None
        elif hasattr(data_cfg, "test"):
            train_set = None
            val_set = None
            test_set = RefineDataset(
                        data_root=data_cfg.test.data_root,
                        pose_path=data_cfg.test.pose_path,
                        rotation_type=data_cfg.test.rotation_type,
                        sample_in_pose = data_cfg.test.sample_in_pose,
                        norm_type=data_cfg.test.norm_type,
                        guidance_type=data_cfg.test.guidance_type,
                        is_train=False,
            )
        return train_set, val_set, test_set
    else:
        raise NotImplementedError(f"Unable to build {data_cfg.name} dataset")
