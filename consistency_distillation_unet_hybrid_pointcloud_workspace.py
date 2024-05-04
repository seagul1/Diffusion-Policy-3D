if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
from diffusion_policy_3d.policy.diffusion_unet_hybrid_pointcloud_policy import DiffusionUnetHybridPointcloudPolicy
from diffusion_policy_3d.dataset.base_dataset import BasePointcloudDataset
from diffusion_policy_3d.env_runner.base_pointcloud_runner import BasePointcloudRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.json_logger import JsonLogger
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler

from diffusion_policy_3d.consistency.consistency_util import create_ema_and_scales_fn, get_weightings,update_ema
from diffusion_policy_3d.consistency.consistent_distillation import ConsistentUnetPointcloudPolicy
from einops import reduce
from torch.nn.modules.batchnorm import _BatchNorm

OmegaConf.register_new_resolver("eval", eval, replace=True)

# todo: rewrite
class TrainConsistencyUnetHybridPointcloudWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        # self.model: DiffusionUnetHybridPointcloudPolicy = hydra.utils.instantiate(cfg.policy)
        self.model: ConsistentUnetPointcloudPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: ConsistentUnetPointcloudPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        
        # --------- consistency function ---------
        self.teacher_model = None
        self.distillation = cfg.distillation
        
        

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        # RUN_VALIDATION = True
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BasePointcloudDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        # dataset element: {'obs', 'action'}
        # obs: {'point_cloud': (T,512,3), 'imagin_robot': (T,96,7), 'agent_pos': (T,D_pos)}
        assert isinstance(dataset, BasePointcloudDataset), print(f"dataset must be BasePointcloudDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
            
        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BasePointcloudRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, BasePointcloudRunner)
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        
        # ------------------------------------------------------------------------
        self.ema_model.requires_grad_(False)
        self.ema_model.train()
        self.ema_model.distillation = False
        
        if self.distillation:
            teacher_policy = hydra.utils.instantiate(cfg.teacher_policy)
            task_name = cfg.task_name
            teacher_policy.load_state_dict(torch.load(f'data/{task_name}.pth'))
            # teacher_policy.load_state_dict(torch.load(f'3D-Diffusion-Policy/data/{task_name}.pth'))
            teacher_policy.to(device)
            teacher_policy.requires_grad_(False)
        
        for dst, src in zip(self.model.model.parameters(), teacher_policy.model.parameters()):
            dst.data.copy_(src.data)
        for dst, src in zip(self.model.obs_encoder.parameters(), teacher_policy.obs_encoder.parameters()):
            dst.data.copy_(src.data)
        # ------------------------------------------------------------------------

        # save batch for sampling
        train_sampling_batch = None
        
        # ---------------------------------
        ema_and_scale_fn = create_ema_and_scales_fn(cfg.target_ema_mode,
                                            cfg.start_ema, 
                                            cfg.scale_mode,
                                            cfg.start_scale,
                                            cfg.end_scale,
                                            cfg.training.num_epochs,
                                            cfg.distill_steps_per_iter)
        # ---------------------------------
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                
                target_ema, scale = ema_and_scale_fn(local_epoch_idx)
                # cprint(f"ema_rate: {target_ema}, num_scale: {scale}", "red")
                
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        t1 = time.time()
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        t1_1 = time.time()
                        
                        # target_ema, scale = ema_and_scale_fn(local_epoch_idx)
                        raw_loss, loss_dict = self.model.consistency_distillation_loss(batch, scale, teacher_model=teacher_policy, target_model=self.ema_model)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        
                        t1_2 = time.time()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        t1_3 = time.time()
                        # update ema
                        # if cfg.training.use_ema:
                        #     update(self.ema_model.parameters(), self.model.parameters(), target_ema)
                        
                        # for param, target_param in zip(self.model.parameters(), self.ema_model.parameters()):
                        #     target_param.data.copy_((1-target_ema) * param.data + target_ema * target_param.data)
                        
                        for module, target_module in zip(self.model.modules(), self.ema_model.modules()):
                            for param, target_param in zip(module.parameters(recurse=False), target_module.parameters(recurse=False)):
                                # iterative over immediate parameters only.
                                if isinstance(param, dict):
                                    raise RuntimeError('Dict parameter not supported')

                                if isinstance(module, _BatchNorm):
                                    # skip batchnorms
                                    target_param.copy_(param.to(dtype=target_param.dtype).data)
                                elif not param.requires_grad:
                                    target_param.copy_(param.to(dtype=target_param.dtype).data)
                                else:
                                    target_param.mul_(target_ema)
                                    target_param.add_(param.data.to(dtype=target_param.dtype), alpha=1 - target_ema)
                                    # target_param.data.copy_((1-target_ema) * param.data + target_ema * target_param.data)
                        
                        t1_4 = time.time()
                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        t1_5 = time.time()
                        step_log.update(loss_dict)
                        t2 = time.time()
                        
                        if verbose:
                            print(f"total one step time: {t2-t1:.3f}")
                            print(f" compute loss time: {t1_2-t1_1:.3f}")
                            print(f" step optimizer time: {t1_3-t1_2:.3f}")
                            print(f" update ema time: {t1_4-t1_3:.3f}")
                            print(f" logging time: {t1_5-t1_4:.3f}")

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                
                if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                    t3 = time.time()
                    # runner_log = env_runner.run(policy, dataset=dataset)
                    runner_log = env_runner.run(policy)
                    t4 = time.time()
                    # print(f"rollout time: {t4-t3:.3f}")
                    # log all
                    step_log.update(runner_log)

                
                    
                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss, loss_dict = self.model.consistency_distillation_loss(batch, scale, teacher_model=teacher_policy, target_model=self.ema_model)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # cprint(f"validation loss: {val_loss}", "red")
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if env_runner is None:
                    step_log['test_mean_score'] = - train_loss
                    
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                del step_log
        

    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure env
        env_runner: BasePointcloudRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BasePointcloudRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
        

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)

def main(cfg):
    workspace = TrainConsistencyUnetHybridPointcloudWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
