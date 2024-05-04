import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import math
import copy
from typing import Dict
from tqdm import tqdm, trange
from copy import deepcopy
from termcolor import cprint

from diffusers import DDPMScheduler
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.policy.base_pointcloud_policy import BasePointcloudPolicy
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator, PointcloudMaskGenerator
import diffusion_policy_3d.model.vision.crop_randomizer as dmvc
from diffusion_policy_3d.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision_3d.pointnet_extractor import DP3Encoder
# from diffusion_policy_3d.model.vision_3d import loss as vision3d_loss
from diffusion_policy_3d.model.vision_3d.se3_aug import create_se3_augmentation

from diffusion_policy_3d.consistency.consistency_util import create_ema_and_scales_fn, get_weightings, append_dims
from diffusers import UNet2DConditionModel

class ConsistentUnetPointcloudPolicy(BasePointcloudPolicy):
    def __init__(self, 
                shape_meta: dict,
                noise_scheduler: DDPMScheduler,
                horizon, 
                n_action_steps, 
                n_obs_steps,
                num_inference_steps=None,
                obs_as_global_cond=True,
                diffusion_step_embed_dim=256,
                down_dims=(256,512,1024),
                kernel_size=5,
                n_groups=8,
                condition_type="film",
                use_down_condition=True,
                use_mid_condition=True,
                use_up_condition=True,
                # ------ condition encode ------
                encoder_output_dim=256,
                crop_shape=None,
                use_pc_color=False,
                pointnet_type="pointnet",
                se3_augmentation_cfg=None,
                pointcloud_encoder_cfg=None,
                # parameters passed to step

                # ------ consistency parameter ------
                distillation=False,
                sigma_min=0.0002,
                sigma_max=80,
                sigma_data=0.5,
                rho=7,
                weight_schedule="karras",
                loss_norm="lpips",

                **kwargs) -> None:
        super().__init__()

        self.condition_type = condition_type

        # parse shape meta
        """
        shape meta: {"action": {"shape": [action_dim]([24])},
                     "obs": {
                             "agent_pos": {"shape": [action_dim]([24]), "type": "low_dim"},
                             "point_cloud": {"shape": [num_points, point_dim]([512, 3]), "type": "point_cloud"}
                             }
                    }
        """
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # point cloud encoder
        obs_encoder = DP3Encoder(observation_space=obs_dict, 
                                img_crop_shape=crop_shape,
                                out_channel=encoder_output_dim,
                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                use_pc_color=use_pc_color,
                                pointnet_type=pointnet_type)
        
        # create diffusion model(consistent model)
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[ConsistencyUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[ConsistencyHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")
        self.se3_augmentation_cfg = se3_augmentation_cfg
        if self.se3_augmentation_cfg.use_aug:
            self.se3_aug = create_se3_augmentation(self.se3_augmentation_cfg)
        else:
            self.se3_aug = None
        cprint(f"[ConsistencyUnetHybridPointcloudPolicy] use_pc_aug: {self.se3_augmentation_cfg.use_aug}", "yellow")

        """
        diffusion: episilon(xt, t)
        consistency: F_theta(x, t)
        """
        model = ConditionalUnet1D(input_dim=input_dim,
                                  local_cond_dim=None,
                                  global_cond_dim=global_cond_dim,
                                  diffusion_step_embed_dim=diffusion_step_embed_dim,
                                  down_dims=down_dims,
                                  kernel_size=kernel_size,
                                  n_groups=n_groups,
                                  condition_type=condition_type,
                                  use_down_condition=use_down_condition,
                                  use_mid_condition=use_mid_condition,
                                  use_up_condition=use_up_condition
                                  )
        
        self.obs_encoder = obs_encoder
        self.model = model
        # todo: 这里的scheduler可能要改
        self.noise_scheduler = noise_scheduler

        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        # # -------------- consistency parameters --------------
        # self.distillation = distillation
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        # self.weight_schedule = weight_schedule
        # self.loss_norm = loss_norm
        self.sigma_data = 0.5


    # ==================== Inference ====================
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs):

            infer_step = 5
        
            model = self.model

            # initial noisy trajectory xT ~ N(0, T^2 * I)
            trajectory = torch.randn(
                size=condition_data.shape, 
                dtype=condition_data.dtype,
                device=condition_data.device) * self.sigma_max

            # set step values
            t_seq = np.linspace(start=self.sigma_min,
                                stop=self.sigma_max,
                                num=infer_step)
            t_seq = torch.from_numpy(t_seq)
            
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(sample=trajectory,
                            timestep=self.sigma_max, 
                            local_cond=local_cond, global_cond=global_cond)
            
            dims = trajectory.ndim
            c_skip_t, c_out_t = self.get_scalings_for_boundary_condition(sigma=self.sigma_max)
            c_skip_t = torch.tensor(c_skip_t, device=trajectory.device)
            c_out_t = torch.tensor(c_out_t, device=trajectory.device)
            trajectory = append_dims(c_skip_t, dims) * trajectory + append_dims(c_out_t, dims) * model_output
            
            for t in t_seq[1:]:
                t = torch.tensor(t, dtype=torch.float32)
                z = torch.rand_like(trajectory)
                
                trajectory = trajectory + math.sqrt(t**2 - self.sigma_min**2) * z
                trajectory[condition_mask] = condition_data[condition_mask]
                
                c_skip_t, c_out_t = self.get_scalings_for_boundary_condition(sigma=t)
                c_skip_t = torch.tensor(c_skip_t, device=trajectory.device)
                c_out_t = torch.tensor(c_out_t, device=trajectory.device)
                model_output = model(sample=trajectory,
                            timestep=t, 
                            local_cond=local_cond, global_cond=global_cond)
                trajectory = append_dims(c_skip_t, dims) * trajectory + append_dims(c_out_t, dims) * model_output
                
            trajectory[condition_mask] = condition_data[condition_mask]  

            return trajectory
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        ----------------------------------------
        obs_dict{"point_cloud": torch.szie([1, 2, 512, 6]),
                 "agent_pos": torch.size([1, 2, 24])}
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            # this_nobs = {"point_cloud": torch.size[2, 512, 3],
            #              "agent_pos": torch.size[2, 24]}
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            #for key, value in this_nobs.items():
                #cprint(f"{key}:{value.shape}", "red")
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result
    

    # ==================== get f_theta ====================
    # EDM method
    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    # Modified EDM method 
    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        # c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out
    
    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas


# ==================== Training ====================
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # def consistency_training_loss(self, batch, num_scale):

    #     # ---------------------- process condition data ----------------------
    #     assert "valid_mask" not in batch
    #     # normalize input
    #     nobs = self.normalizer(batch["obs"])
    #     nactions = self.normalizer["action"].normalize(batch["action"])

    #     if not self.use_pc_color:
    #         nobs['point_cloud'] = nobs['point_cloud'][..., :3]
    #     if self.se3_aug is not None:
    #         # Point cloud: batch_size, horizon, num_points, dim_point
    #         B, T, N, D = nobs["point_cloud"].shape
    #         nobs["point_cloud"] = self.se3_aug(nobs["point_cloud"].reshape(B*T, N, D)).reshape(B, T, N, D)
        
    #     batch_size = nactions.shape[0]
    #     horizon = nactions.shape[1]

    #     #handle differrent ways of passing observation
    #     local_cond = None
    #     global_cond = None
    #     trajectory = nactions
    #     cond_data = trajectory

    #     if self.obs_as_global_cond:
    #         # reshape B, T, ... to B*T
    #         this_nobs = dict_apply(nobs, 
    #             lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
    #         nobs_features = self.obs_encoder(this_nobs)

    #         if "cross_attention" in self.condition_type:
    #             # treat as sequence
    #             global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
    #         else:
    #             # reshape back to B, Do
    #             global_cond = nobs_features.reshape(batch_size, -1)

    #         this_n_point_cloud = this_nobs["point_cloud"].reshape(batch_size, -1, *this_nobs["point_cloud"].shape[1:])
    #         this_n_point_cloud = this_n_point_cloud[..., :3]
    #     else:
    #         # reshape B, T, .. to B*T
    #         this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
    #         nobs_features = self.obs_encoder(this_nobs)
    #         # reshape back to B, T, Do
    #         nobs_features = nobs_features.reshape(batch_size, horizon, -1)
    #         cond_data = torch.cat([nactions, nobs_features], dim=-1)
    #         trajectory = cond_data.detach()

    #     # generate impainting mask
    #     condition_mask = self.mask_generator(trajectory.shape)

    #     # ---------------------- compute consistency training loss ----------------------
    #     # n ~ Uniform[1, N(k) - 1], num_scale = N(epoch) N(.) represents step schedule
    #     # N(k) = ceiling(((k/K) * ((s_1 + 1)^2) + s_0^2) - 1) + 1
    #     num_steps = torch.randint(1, num_scale, (batch_size))
        
    #     # indces: torch.szie(128,)
    #     indices = torch.randint(
    #         0, num_steps - 1, (trajectory.shape[0],), device=trajectory.device
    #     )

    #     noise = torch.randn_like(trajectory, device=trajectory.device)

    #     # t_i = (ϵ ** (1/ρ) + (i-1)/(N-1) * (T ** (1/ρ) - ϵ ** (1/ρ)) ** ρ
    #     t = self.sigma_max ** (1 / self.rho) + indices / (num_steps - 1) * (
    #         self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
    #     )
    #     t = t**self.rho

    #     t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_steps - 1) * (
    #         self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
    #     )
    #     t2 = t2**self.rho

    #     noise_trajectory = trajectory + t * noise
    #     pass

    def consistency_training_loss(self, batch, t, noise):

        # ---------------------- process condition data ----------------------
        assert "valid_mask" not in batch
        # normalize input
        nobs = self.normalizer(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        if self.se3_aug is not None:
            # Point cloud: batch_size, horizon, num_points, dim_point
            B, T, N, D = nobs["point_cloud"].shape
            nobs["point_cloud"] = self.se3_aug(nobs["point_cloud"].reshape(B*T, N, D)).reshape(B, T, N, D)
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        #handle differrent ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)

            this_n_point_cloud = this_nobs["point_cloud"].reshape(batch_size, -1, *this_nobs["point_cloud"].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, .. to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # ---------------------- compute consistency f_theta(x_t, t) ----------------------
        # noise = torch.randn_like(trajectory, device=trajectory.device)

        assert noise.shape[0] == t.shape[0]
        dims = noise.ndim
        noisy_trajectory = trajectory + append_dims(t, dims) * noise
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
                # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=t, 
                            local_cond=local_cond, 
                            global_cond=global_cond)
        
        c_skip_t, c_out_t = self.get_scalings_for_boundary_condition(sigma=t)
        f_theta = append_dims(c_skip_t, dims) * trajectory + append_dims(c_out_t, dims) * pred
        return f_theta
