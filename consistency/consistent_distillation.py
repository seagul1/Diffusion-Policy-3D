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

from diffusers import DDPMScheduler, KarrasVeScheduler
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

from diffusion_policy_3d.consistency.consistency_util import mean_flat, get_weightings, append_dims, get_sigmas_karras, sample_heun, DummyGenerator


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
                distillation=True,
                sigma_min=0.002,
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
        self.num_inference_steps = num_inference_steps

        # -------------- consistency parameters --------------
        self.distillation = distillation
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.weight_schedule = weight_schedule
        self.loss_norm = loss_norm
        self.sigma_data = sigma_data


    # ==================== Inference ====================
    def conditional_sample(self, steps,
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs):
        sigmas = get_sigmas_karras(steps, self.sigma_min, self.sigma_max, self.rho, device="cuda:0")
        
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device) 
        # 要不要调换顺序
        trajectory[condition_mask] = condition_data[condition_mask]
        trajectory = trajectory * append_dims(torch.as_tensor(self.sigma_max, device=condition_data.device), condition_data.ndim)

        _, trajectory = self.denoise(trajectory, self.sigma_max, global_cond, local_cond)
        
        for i in range(steps - 1):
            noise = torch.rand_like(trajectory)
            trajectory[condition_mask] = condition_data[condition_mask]
            trajectory = trajectory + math.sqrt(sigmas[i]**2 - self.sigma_min**2) * noise
            
            _, trajectory = self.denoise(trajectory, sigmas[i], global_cond, local_cond)
        
        # generator = DummyGenerator()
        # trajectory = generator.randn(*condition_data.shape, device=condition_data.device) * self.sigma_max
        
        # def denoiser(trajectory, sigma, global_cond, local_cond):
        #     _, denoised = self.denoise(trajectory, sigma, global_cond, local_cond)
        #     return denoised
            
        # trajectory = sample_heun(
        #     denoiser,
        #     trajectory,
        #     sigmas,
        #     generator,
        #     condition_data=condition_data,
        #     condition_mask=condition_mask,
        #     local_cond=local_cond,
        #     global_cond=global_cond
        #     )
            
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
        # ----------------------------------------------
        # 最好： 5
        # steps = self.num_inference_steps
        steps = 5
        # ----------------------------------------------
        nsample = self.conditional_sample(steps,
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
    

    # ==================== get f_theta parameters ====================
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

    def denoise(self, x_t, sigmas, global_cond, local_cond, model=None):
        if model is None:
            unet_model = self.model
        else:
            unet_model = model.model
        # c_skip, c_out,c_in = [
        #     append_dims(torch.tensor(x, device=x_t.device), x_t.ndim)
        #     for x in self.get_scalings_for_boundary_condition(sigmas)
        # ]
        c_skip, c_out = [
            append_dims(torch.as_tensor(x, device=x_t.device), x_t.ndim)
            for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        model_output = unet_model(sample=x_t,
                                  timestep=sigmas,
                                  local_cond=local_cond,
                                  global_cond=global_cond)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised

# ==================== Training ====================
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        
    
    def process_condition(self, batch, model=None):
        if model is None:
            obs_encoder = self.obs_encoder
        else:
            obs_encoder = model.obs_encoder
        # ---------------------- process condition data ----------------------
        assert "valid_mask" not in batch
        # normalize input
        # print("pc before min, max, mean:", batch['obs']['point_cloud'][...,:3].max(), batch['obs']['point_cloud'][...,:3].min(), batch['obs']['point_cloud'][...,:3].mean())
        # print("agent pos before min, max, mean:", batch['obs']['agent_pos'].max(), batch['obs']['agent_pos'].min(), batch['obs']['agent_pos'].mean())
        nobs = self.normalizer.normalize(batch['obs'])
        # print("pc after min, max, mean:", nobs['point_cloud'][...,:3].max(), nobs['point_cloud'][...,:3].min(), nobs['point_cloud'][...,:3].mean())
        # print("agent pos after min, max, mean:", nobs['agent_pos'].max(), nobs['agent_pos'].min(), nobs['agent_pos'].mean())
        # nobs = self.normalizer(batch["obs"])
        
        # print("action before, min, max, mean:", batch['action'].min(), batch['action'].max(), batch['action'].mean())
        nactions = self.normalizer["action"].normalize(batch["action"])
        # print("action after, min, max, mean:", nactions.min(), nactions.max(), nactions.mean())

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
            nobs_features = obs_encoder(this_nobs)

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
            nobs_features = obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        return trajectory, cond_data, condition_mask, global_cond, local_cond
        

    def consistency_distillation_loss(self, batch, num_scales, teacher_model=None, target_model=None):
        
        self.obs_encoder.requires_grad_(False)
        
        trajectory, cond_data, condition_mask, global_cond, local_cond = self.process_condition(batch)
        
        # if self.distillation:
        #     if teacher_model is None:
        #         raise ValueError("teacher model is None")
        #     else:
        #         _, _, _, teac_global_cond, _ = self.process_condition(batch, model=teacher_model)
        
        

        # ---------------------- compute consistency training loss ----------------------
        # 1. sample indices ~ Uniform[1, N(.) - 1]
        indices = torch.randint(
            0, num_scales - 1, (trajectory.shape[0],), device=trajectory.device
        )

        # t_i = (ϵ ** (1/ρ) + (i-1)/(N-1) * (T ** (1/ρ) - ϵ ** (1/ρ)) ** ρ
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        # t = self.sigma_min ** (1 / self.rho) + indices / (num_scales - 1) * (
        #     self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho)
        # )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        # t2 = self.sigma_min ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
        #     self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho)
        # )
        t2 = t2**self.rho
        
        # 2. sample x_t_{n+1} ~ N(x; t_{n+1}^2 * I) and apply condition
        # sample noise z ~ N(0, I)
        noise = torch.randn_like(trajectory, device=trajectory.device)

        noisy_trajectory = trajectory + append_dims(t, trajectory.ndim) * noise
        
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        @torch.no_grad()
        def euler_solver(samples, global_cond, local_cond, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                _, denoiser = self.denoise(x, t, global_cond, local_cond, model=teacher_model)
            d = (x - denoiser) / append_dims(t, x.ndim)
            samples = x + d * append_dims(next_t - t, x.ndim)

            return samples
        
        @torch.no_grad()
        def heun_solver(samples, global_cond, local_cond, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                # _, denoiser = self.denoise(x, t, global_cond, local_cond, model=teacher_model)
                denoiser = teacher_model.model(x, t, global_cond=global_cond, local_cond=local_cond)

            d = (x - denoiser) / append_dims(t, x.ndim)
            samples = x + d * append_dims(next_t - t, x.ndim)
            if teacher_model is None:
                denoiser = x0
            else:
                # _, denoiser = self.denoise(x, next_t, global_cond, local_cond, model=teacher_model)
                denoiser = teacher_model.model(x, next_t, global_cond=global_cond, local_cond=local_cond)

            next_d = (samples - denoiser) / append_dims(next_t, x.ndim)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, x.ndim)
            return samples
        
        dropout_state = torch.get_rng_state()
        # f_theta(x_{t_{n+1}}, t_{n+1})
        _, distiller = self.denoise(noisy_trajectory, t, global_cond, local_cond)

        # x_tn_fi = x_{t_{n+1}} + (t_n - t_{n+1}) * fi(x_{t_{n+1}}, t_{n+1})
        # x_t2 = euler_solver(noisy_trajectory, teac_global_cond, local_cond, t, t2, trajectory).detach()
        # x_t2 = heun_solver(noisy_trajectory, teac_global_cond, local_cond, t, t2, trajectory).detach()
        # 2024/04/17 modify
        x_t2 = heun_solver(noisy_trajectory, global_cond, local_cond, t, t2, trajectory).detach()
        x_t2[condition_mask] = cond_data[condition_mask]


        _, _, _, tar_global_cond, _ = self.process_condition(batch, model=target_model)
        torch.set_rng_state(dropout_state)
        # f_theta_ema(x_tn_fi, tn)
        # x_t2[tar_cond_mask] = tar_cond_data[tar_cond_data]
        _, distiller_target = self.denoise(x_t2, t2, global_cond, local_cond, model=target_model)
        distiller_target = distiller_target.detach()
        
        snrs = self.get_snr(sigmas=t2)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        
        # loss = F.mse_loss(distiller, distiller_target, reduction="none")
        # loss = loss * append_dims(weights, loss.ndim)
        # loss = loss * loss_mask.type(loss.dtype)
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss.mean()
        
        if self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            # diffs = ((distiller - distiller_target) ** 2 + (distiller - trajectory) ** 2 ) / 2
            loss = mean_flat(diffs) * weights
            # loss = mean_flat(diffs)
            loss = loss.mean()
            
        else:
            raise NotImplementedError(f"loss {self.loss_norm} have not been implemented!") 
            
        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict
        



