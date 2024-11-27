from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from .decoder import build_decoder
from .loss import GraspLossPose
from .utils.diffusion_utils import make_schedule_ddpm

class DDPM(nn.Module):
    def __init__(self, cfg) -> None:
        super(DDPM, self).__init__()
        
        self.eps_model = build_decoder(cfg.decoder)
        self.criterion = GraspLossPose(cfg.criterion)
        self.loss_weights = cfg.criterion.loss_weights
        self.timesteps = cfg.steps
        self.schedule_cfg = cfg.schedule_cfg
        self.rand_t_type = cfg.rand_t_type
        self.pred_x0 = cfg.pred_x0

        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)

    @property
    def device(self):
        return self.betas.device
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ Forward difussion process, $q(x_t \mid x_0)$, this process is determinative 
        and has no learnable parameters.

        $x_t = \sqrt{\bar{\alpha}_t} * x0 + \sqrt{1 - \bar{\alpha}_t} * \epsilon$

        Args:
            x0: samples at step 0
            t: diffusion step
            noise: Gaussian noise
        
        Return:
            Diffused samples
        """
        B, *x_shape = x0.shape
        x_t = self.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x0 + \
            self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise

        return x_t

    def forward(self, data: Dict):
        if self.training:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    def forward_train(self, data: Dict):
        """ Reverse diffusion process, sampling with the given data containing condition

        Args:
            data: test data, data['norm_pose'] gives the target data, 
            data['obj_pc'] gives the condition
        
        Return:
            Computed loss
        """
        B = data['norm_pose'].shape[0]

        ## randomly sample timesteps
        if self.rand_t_type == 'all':
            ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()
        elif self.rand_t_type == 'half':
            ts = torch.randint(0, self.timesteps, ((B + 1) // 2, ), device=self.device)
            if B % 2 == 1:
                ts = torch.cat([ts, self.timesteps - ts[:-1] - 1], dim=0).long()
            else:
                ts = torch.cat([ts, self.timesteps - ts - 1], dim=0).long()
        else:
            raise Exception('Unsupported rand ts type.')
        
        ## generate Gaussian noise
        noise = torch.randn_like(data['norm_pose'], device=self.device)

        ## calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=data['norm_pose'], t=ts, noise=noise)

        ## predict noise
        data["cond_obj"] = self.eps_model.condition_obj(data)
        data["cond_txt_cls"], data["cond_txt"] = self.eps_model.condition_language(data)

        output = self.eps_model(x_t, ts, data)

        if self.pred_x0:
            pred_dict = {
                "pred_pose_norm": output,
                "noise": noise
            }
        else:
            B, *x_shape = x_t.shape
            pred_x0 = self.sqrt_recip_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * x_t - \
                        self.sqrt_recipm1_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * output
            pred_dict = {
                "pred_noise": output,
                "pred_pose_norm": pred_x0,
                "noise": noise,
            }

        loss_dict = self.criterion(pred_dict, data)


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
        pred_x0 = self.sample(data)
        pred_x0 = pred_x0[:,0,-1]

        pred_dict = {
                "pred_pose_norm": pred_x0,
                "qpos_norm": pred_x0[..., 3:25],
                "translation_norm": pred_x0[..., :3],
                "rotation": pred_x0[..., 25:],
                "pred_noise": torch.tensor([1.0]),
                "noise": torch.tensor([1.0])
            }
        
        loss_dict, preditions = self.criterion(pred_dict, data)

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
        pred_x0 = self.sample(data, k=k)
        # [1, 4, 101, 28]
        pred_x0 = pred_x0[:, :, -1]

        pred_dict = {
                "pred_pose_norm": pred_x0,
                "qpos_norm": pred_x0[..., 3:25],
                "translation_norm": pred_x0[..., :3],
                "rotation": pred_x0[..., 25:],
                "pred_noise": torch.tensor([1.0]),
                "noise": torch.tensor([1.0])
            }
        
        preds_hand, targets_hand = self.criterion.infer_norm_process_dict(pred_dict, data)

        return preds_hand, targets_hand, 
    
    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, data: Dict) -> Tuple:
        """ Get and process model prediction

        $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            The predict target `(pred_noise, pred_x0)`, currently we predict the noise, which is as same as DDPM
        """
        B, *x_shape = x_t.shape
        if self.pred_x0:
            pred_x0 = self.eps_model(x_t, t, data)
            # pred_x0 = torch.clamp(pred_x0, -1, 1)
            pred_noise = (self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - pred_x0) \
                            / self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape)))
        else:
            pred_noise = self.eps_model(x_t, t, data)
            pred_x0 = self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
                        self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * pred_noise

        return pred_noise, pred_x0
    
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, data:Dict) -> Tuple:
        """ Calculate the mean and variance, we adopt the following first equation.

        $\tilde{\mu} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0$
        $\tilde{\mu} = \frac{1}{\sqrt{\alpha}_t}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            (model_mean, posterior_variance, posterior_log_variance)
        """
        B, *x_shape = x_t.shape

        ## predict noise and x0 with model $p_\theta$
        pred_noise, pred_x0 = self.model_predict(x_t, t, data)

        ## calculate mean and variance
        model_mean = self.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * x_t
        posterior_variance = self.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, data: Dict) -> torch.Tensor:
        """ One step of reverse diffusion process

        $x_{t-1} = \tilde{\mu} + \sqrt{\tilde{\beta}} * z$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            data: data dict that provides original data and computed conditional feature

        Return:
            Predict data in the previous step, i.e., $x_{t-1}$
        """
        B, *_ = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, data)
        
        noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0

        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_x
    
    @torch.no_grad()
    def p_sample_loop(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['norm_pose'] gives the target data shape
        
        Return:
            Sampled data, <B, T, ...>
        """
        if isinstance(data['norm_pose'], Tensor) and data['norm_pose'].dim()==2:
            x_t = torch.randn_like(data['norm_pose'], device=self.device)
        else:
            x_t = torch.randn(len(data['norm_pose']), data['norm_pose'][0].shape[-1], device=self.device)
        ## precompute conditional feature, which will be used in every sampling step
        data["cond_obj"] = self.eps_model.condition_obj(data)
        data["cond_txt_cls"], data["cond_txt"] = self.eps_model.condition_language(data)

        ## iteratively sampling
        all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_sample(x_t, t, data)
            all_x_t.append(x_t)
        return torch.stack(all_x_t, dim=1)
    
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
    
