from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from .decoder import build_decoder
from .loss import GraspLossPose
from .utils.diffusion_utils import make_schedule_ddpm

class RenfinmentTransformer(nn.Module):
    def __init__(self, cfg) -> None:
        super(RenfinmentTransformer, self).__init__()
        
        self.eps_model = build_decoder(cfg.decoder)
        self.criterion = GraspLossPose(cfg.criterion)
        self.loss_weights = cfg.criterion.loss_weights
        self.timesteps = cfg.steps
        self.schedule_cfg = cfg.schedule_cfg
        self.rand_t_type = cfg.rand_t_type
        self.pred_abs = cfg.pred_abs

        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)

    @property
    def device(self):
        return self.betas.device


    def forward(self, data: Dict):
        """ Reverse diffusion process, sampling with the given data containing condition

        Args:
            data: test data, data['norm_pose'] gives the target data, 
            data['obj_pc'] gives the condition
        
        Return:
            Computed loss
        """
        B = data['coarse_norm_pose'].shape[0]

        ## predict noise
        data['hand_pc'] = self.criterion.hand_model(data['coarse_pose'], with_surface_points=True)["surface_points"]
        data['cond_hand'] = self.eps_model.condition_hand(data)

        data["cond_obj"] = self.eps_model.condition_obj(data)

        output = self.eps_model(data['coarse_norm_pose'], None, data)

        if not self.pred_abs:
            output += data['coarse_norm_pose']
        if self.training:
            loss_dict = self.criterion({"pred_pose_norm": output}, data)
        else:
            loss_dict, _e = self.criterion({"pred_pose_norm": output}, data)

        loss = 0
        for k, v in loss_dict.items():
            if k in self.loss_weights:
                loss += v * self.loss_weights[k]

        return loss, loss_dict, None

    def forward_test(self, data: Dict):
        """ Reverse diffusion process, sampling with the given data containing condition

        Args:
            data: test data, data['norm_pose'] gives the target data, 
            data['obj_pc'] gives the condition
        
        Return:
            Computed loss
        """
        B = data['coarse_norm_pose'].shape[0]

        ## predict noise
        data['hand_pc'] = self.criterion.hand_model(data['coarse_pose'], with_surface_points=True)["surface_points"]
        data['cond_hand'] = self.eps_model.condition_hand(data)

        data["cond_obj"] = self.eps_model.condition_obj(data)

        output = self.eps_model(data['coarse_norm_pose'], None, data)

        if not self.pred_abs:
            output += data['coarse_norm_pose']
        if self.training:
            loss_dict = self.criterion({"pred_pose_norm": output}, data)
        else:
            loss_dict, preditions = self.criterion({"pred_pose_norm": output}, data)

        loss = 0
        for k, v in loss_dict.items():
            if k in self.loss_weights:
                loss += v * self.loss_weights[k]

        return loss, loss_dict, preditions
    
    def forward_infer(self, data: Dict, k=4):
        """ Reverse diffusion process, sampling with the given data containing condition

        Args:
            data: test data, data['norm_pose'] gives the target data, 
            data['obj_pc'] gives the condition
        
        Return:
            Computed loss
        """
        B = data['coarse_norm_pose'].shape[0]

        ## predict noise
        data['hand_pc'] = self.criterion.hand_model(data['coarse_pose'], with_surface_points=True)["surface_points"]
        data['cond_hand'] = self.eps_model.condition_hand(data)

        data["cond_obj"] = self.eps_model.condition_obj(data)

        output = self.eps_model(data['coarse_norm_pose'], None, data)

        if not self.pred_abs:
            output += data['coarse_norm_pose']

        preds_hand, targets_hand = self.criterion.infer_norm_process_dict({"pred_pose_norm": output}, data)

        return preds_hand, targets_hand
    

    @torch.no_grad()
    def sample(self, data: Dict, k: int=1) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition
        In this method, the sampled results are unnormalized and converted to absolute representation.

        Args:
            data: test data, data['norm_pose'] gives the target data shape
            k: the number of sampled data
        
        Return:
            Sampled results, the shape is <B, k, T, ...>
        """
        ## TODO ddim sample function
        ksamples = []
        for _ in range(k):
            ksamples.append(self.p_sample_loop(data))
        ksamples = torch.stack(ksamples, dim=1)
        return ksamples
    
