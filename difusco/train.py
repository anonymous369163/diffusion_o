"""The handler for training and evaluation."""

import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info
import pytorch_lightning as pl

from pl_tsp_model import TSPModel
from pl_mis_model import MISModel


class GradientLoggingCallback(pl.Callback):
    """记录梯度范数等重要训练信息的回调函数"""
    
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
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
  parser.add_argument('--inference_schedule', type=str, default='linear')
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

  parser.add_argument('--do_train', action='store_true', default=False)
  parser.add_argument('--do_test', action='store_true', default=True)
  parser.add_argument('--do_valid_only', action='store_true')
  parser.add_argument('--rl_compute_frequency', type=int, default=1)
  parser.add_argument('--use_pomo', action='store_true', default=True)
  parser.add_argument('--rl_loss_weight', type=float, default=0.01)
  parser.add_argument('--rl_baseline_decay', type=float, default=0.95)
  parser.add_argument('--pomo_temperature', type=float, default=1.0)
  parser.add_argument('--no_debug', action='store_true', default=False)
  parser.add_argument('--problem_type', type=str, default='TSP')
  parser.add_argument('--add_prior', action='store_true', default=False)
  parser.add_argument('--draw_route_comparison', action='store_true', default=False)
  parser.add_argument('--rl_debug', action='store_true', default=False)
  parser.add_argument('--test_debug', action='store_true', default=False)
  
  # 添加新的参数用于指定混合训练的问题类型
  parser.add_argument('--hybrid_problem_types', type=str, nargs='+', default=None,
                      help='List of problem types for hybrid training (e.g., CVRP VRPTW OVRP)')

  args = parser.parse_args()
  return args


def generate_hybrid_configs(problem_types):
  """根据指定的问题类型生成混合训练配置"""
  # 问题类型到数据文件的映射
  problem_to_file = {
      'CVRP': './data/vrp/CVRP50.pkl',
      'VRPTW': './data/vrp/VRPTW50.pkl',
      'OVRP': './data/vrp/OVRP50.pkl',
      'VRPB': './data/vrp/VRPB50.pkl',
      'VRPL': './data/vrp/VRPL50.pkl',
      'OVRPTW': './data/vrp/OVRPTW50.pkl',
      'OVRPB': './data/vrp/OVRPB50.pkl',
      'VRPBL': './data/vrp/VRPBL50.pkl',
      'VRPBTW': './data/vrp/VRPBTW50.pkl',
      'VRPLTW': './data/vrp/VRPLTW50.pkl',
      'OVRPBL': './data/vrp/OVRPBL50.pkl',
      'OVRPBTW': './data/vrp/OVRPBTW50.pkl',
      'OVRPLTW': './data/vrp/OVRPLTW50.pkl',
      'VRPBLTW': './data/vrp/VRPBLTW50.pkl',
      'OVRPBLTW': './data/vrp/OVRPBLTW50.pkl'
  }
  
  configs = []
  for problem_type in problem_types:
    if problem_type in problem_to_file:
      configs.append({
          'data_file': problem_to_file[problem_type],
          'problem_type': problem_type
      })
    else:
      print(f"警告：未知的问题类型 {problem_type}，将被忽略")
  
  return configs


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
    model_class = TSPModel
    saving_mode = 'min'

  model = model_class(param_args=args)

  # 根据训练或测试阶段确定版本名称
  tb_save_dir = os.path.join(args.storage_path, 'tb_logs')
  logger_name = args.logger_name or project_name
  
  # 修改：不再手动计算版本号，让TensorBoardLogger自动管理
  if args.do_train:
    # 训练阶段 - 添加train前缀到logger名称
    logger_name_with_phase = f"{logger_name}_train"
  else:
    # 仅测试阶段 - 添加test前缀到logger名称  
    logger_name_with_phase = f"{logger_name}_test"

  # Create TensorBoard logger - 不指定version，让它自动递增
  tb_logger = TensorBoardLogger(
      save_dir=tb_save_dir,
      name=logger_name_with_phase
  )
  rank_zero_info(f"Logging to {tb_logger.log_dir}")  # 直接显示完整的日志目录路径

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
      monitor='val/best_solved_cost', mode=saving_mode,
      save_top_k=3, save_last=True,
      dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),  # 使用log_dir确保路径一致
  )
  lr_callback = LearningRateMonitor(logging_interval='step')
  gradient_callback = GradientLoggingCallback()

  if args.no_debug:  # no_debug表示训练模型，不是debug模式
    trainer = Trainer(
      accelerator="auto",
      devices=torch.cuda.device_count() if torch.cuda.is_available() else 0,  
      max_epochs=epochs,
      callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback, gradient_callback],
      logger=tb_logger,
      check_val_every_n_epoch=1,
      strategy= DDPStrategy(static_graph=True),
      precision=16 if args.fp16 else 32,
  )
  else:
    trainer = Trainer(
        accelerator="auto",
        devices=1 if not args.no_debug else torch.cuda.device_count() if torch.cuda.is_available() else 0,  
        max_epochs=epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback, gradient_callback],
        logger=tb_logger,
        check_val_every_n_epoch=1, 
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
  if args.problem_type == "hybrid":
    # 混合训练配置：根据用户指定的问题类型生成配置
    if args.hybrid_problem_types:
      # 使用用户指定的问题类型
      args.hybrid_configs = generate_hybrid_configs(args.hybrid_problem_types)
      print(f"使用指定的问题类型: {args.hybrid_problem_types}")
    else:
      # 如果没有指定，使用所有支持的问题类型
      all_problem_types = ['CVRP', 'VRPTW', 'OVRP', 'VRPB', 'VRPL', 'OVRPTW', 
                          'OVRPB', 'VRPBL', 'VRPBTW', 'VRPLTW', 'OVRPBL', 
                          'OVRPBTW', 'OVRPLTW', 'VRPBLTW', 'OVRPBLTW']
      args.hybrid_configs = generate_hybrid_configs(all_problem_types)
      print(f"使用所有支持的问题类型: {all_problem_types}")
    
    print(f"生成的混合训练配置包含 {len(args.hybrid_configs)} 个问题类型")

  main(args)
