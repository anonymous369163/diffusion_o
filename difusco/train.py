"""The handler for training and evaluation."""

import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info
import pytorch_lightning as pl

from pl_tsp_model import TSPModel
from pl_mis_model import MISModel


class GradientLoggingCallback(pl.Callback):
    """记录梯度范数等重要训练信息的回调函数"""
    
    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx):
        # 计算梯度范数
        total_norm = 0
        param_count = 0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        total_norm = total_norm ** (1. / 2)
        
        # 记录梯度范数
        if param_count > 0:
            pl_module.log("train/gradient_norm", total_norm, on_step=True, on_epoch=False)
            pl_module.log("train/gradient_norm_avg", total_norm / param_count, on_step=True, on_epoch=False)
    
    def on_train_epoch_start(self, trainer, pl_module):
        # 记录当前epoch的学习率
        if hasattr(trainer.optimizers[0], 'param_groups'):
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            pl_module.log("train/learning_rate", current_lr, on_step=False, on_epoch=True)


def arg_parser():
  parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TSP dataset.')
  parser.add_argument('--task', type=str, default='tsp')
  parser.add_argument('--storage_path', type=str, default='./')
  parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_train_concorde.txt')
  parser.add_argument('--training_split_label_dir', type=str, default=None,
                      help="Directory containing labels for training split (used for MIS).")
  parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--test_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--validation_examples', type=int, default=8)

  parser.add_argument('--batch_size', type=int, default=64)  
  parser.add_argument('--num_epochs', type=int, default=50)
  parser.add_argument('--learning_rate', type=float, default=2e-4)
  parser.add_argument('--weight_decay', type=float, default=1e-4)
  parser.add_argument('--lr_scheduler', type=str, default='cosine-decay')

  parser.add_argument('--num_workers', type=int, default=1)   # o:16
  parser.add_argument('--fp16', action='store_true')
  parser.add_argument('--use_activation_checkpoint', action='store_true')

  parser.add_argument('--diffusion_type', type=str, default='categorical')  # o:gaussian
  parser.add_argument('--diffusion_schedule', type=str, default='cosine')
  parser.add_argument('--diffusion_steps', type=int, default=1000) 
  parser.add_argument('--inference_diffusion_steps', type=int, default=50)  # o:1k
  parser.add_argument('--inference_schedule', type=str, default='cosine')
  parser.add_argument('--inference_trick', type=str, default="ddim")
  parser.add_argument('--sequential_sampling', type=int, default=1)
  parser.add_argument('--parallel_sampling', type=int, default=1)

  parser.add_argument('--n_layers', type=int, default=12)
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--sparse_factor', type=int, default=-1)
  parser.add_argument('--aggregation', type=str, default='sum')
  parser.add_argument('--two_opt_iterations', type=int, default=1000)
  parser.add_argument('--save_numpy_heatmap', action='store_true')

  parser.add_argument('--project_name', type=str, default='tsp_diffusion')
  parser.add_argument('--logger_name', type=str, default=None)
  parser.add_argument('--ckpt_path', type=str, default=None)
  parser.add_argument('--resume_weight_only', action='store_true')

  parser.add_argument('--do_train', action='store_true', default=True)
  parser.add_argument('--do_test', action='store_true', default=True)
  parser.add_argument('--do_valid_only', action='store_true')
  parser.add_argument('--rl_compute_frequency', type=int, default=1)
  parser.add_argument('--use_pomo', action='store_true', default=True)
  parser.add_argument('--rl_loss_weight', type=float, default=0.01)
  parser.add_argument('--rl_baseline_decay', type=float, default=0.95)
  parser.add_argument('--pomo_temperature', type=float, default=1.0)

  args = parser.parse_args()
  return args


def main(args):
  import sys
  # 修改当前工作目录为项目根目录
  current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.chdir(current_dir) 
  epochs = args.num_epochs
  project_name = args.project_name

  if args.task == 'tsp':
    model_class = TSPModel
    saving_mode = 'min'
  elif args.task == 'mis':
    model_class = MISModel
    saving_mode = 'max'
  else:
    raise NotImplementedError

  model = model_class(param_args=args)

  # Create TensorBoard logger
  tb_logger = TensorBoardLogger(
      save_dir=os.path.join(args.storage_path, 'tb_logs'),
      name=args.logger_name or project_name,
  )
  rank_zero_info(f"Logging to {tb_logger.save_dir}/{tb_logger.name}/{tb_logger.version}")

  # 记录重要的训练信息
  if rank_zero_info:
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    # 记录模型和训练信息到tensorboard
    tb_logger.experiment.add_text("model_info/architecture", str(model.model), 0)
    tb_logger.experiment.add_scalar("model_info/total_parameters", total_params, 0)
    tb_logger.experiment.add_scalar("model_info/trainable_parameters", trainable_params, 0)
    
    # 记录数据集信息
    tb_logger.experiment.add_text("dataset_info/training_split", args.training_split, 0)
    tb_logger.experiment.add_text("dataset_info/validation_split", args.validation_split, 0)
    tb_logger.experiment.add_text("dataset_info/test_split", args.test_split, 0)
    
    # 记录训练配置
    training_config = f"""
    Task: {args.task}
    Batch Size: {args.batch_size}
    Learning Rate: {args.learning_rate}
    Weight Decay: {args.weight_decay}
    LR Scheduler: {args.lr_scheduler}
    Epochs: {args.num_epochs}
    FP16: {args.fp16}
    Use Activation Checkpoint: {args.use_activation_checkpoint}
    """
    tb_logger.experiment.add_text("training_config", training_config, 0)
    
    # 记录扩散模型配置
    diffusion_config = f"""
    Diffusion Type: {args.diffusion_type}
    Diffusion Schedule: {args.diffusion_schedule}
    Diffusion Steps: {args.diffusion_steps}
    Inference Steps: {args.inference_diffusion_steps}
    Inference Schedule: {args.inference_schedule}
    Inference Trick: {args.inference_trick}
    Sequential Sampling: {args.sequential_sampling}
    Parallel Sampling: {args.parallel_sampling}
    """
    tb_logger.experiment.add_text("diffusion_config", diffusion_config, 0)

  checkpoint_callback = ModelCheckpoint(
      monitor='val/solved_cost', mode=saving_mode,
      save_top_k=3, save_last=True,
      dirpath=os.path.join(tb_logger.save_dir,
                           tb_logger.name,
                           f'version_{tb_logger.version}',
                           'checkpoints'),
  )
  lr_callback = LearningRateMonitor(logging_interval='step')
  gradient_callback = GradientLoggingCallback()

  trainer = Trainer(
      accelerator="auto",
      devices=torch.cuda.device_count() if torch.cuda.is_available() else None,  
      # devices=1,  
      max_epochs=epochs,
      callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback, gradient_callback],
      logger=tb_logger,
      check_val_every_n_epoch=1,
      strategy=DDPStrategy(static_graph=True), 
      precision=16 if args.fp16 else 32,
  )

  rank_zero_info(
      f"{'-' * 100}\n"
      f"{str(model.model)}\n"
      f"{'-' * 100}\n"
  )

  ckpt_path = args.ckpt_path

  if args.do_train:
    if args.resume_weight_only:
      model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
      trainer.fit(model)
    else:
      trainer.fit(model, ckpt_path=ckpt_path)

    if args.do_test:
      trainer.test(ckpt_path=checkpoint_callback.best_model_path)

  elif args.do_test:
    trainer.validate(model, ckpt_path=ckpt_path)
    if not args.do_valid_only:
      trainer.test(model, ckpt_path=ckpt_path)


if __name__ == '__main__':
  args = arg_parser()
  main(args)
