""" Test the resfusion restore module """
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from datamodule import ISTD_DataModule
from model.denoising_module import RDDM_Unet, DiT_models
from model import GaussianResfusion_Restore_Mask
from variance_scheduler import LinearProScheduler, CosineProScheduler
import torch


def main(args):
    # 设定随机种子以及一些因素，用来控制实验结果
    if args.set_float32_matmul_precision_high:
        torch.set_float32_matmul_precision('high')
    if args.set_float32_matmul_precision_medium:
        torch.set_float32_matmul_precision('medium')

    pl.seed_everything(args.seed, workers=True)

    # 数据集的选择
    if args.dataset == 'ISTD':
        data_module = ISTD_DataModule(root_dir=args.data_dir, batch_size=args.batch_size, pin_mem=args.pin_mem,
                                      num_workers=args.num_workers)
    else:
        raise ValueError("Wrong dataset type !!!")

    # 噪声调度器的选择
    if args.noise_schedule == 'LinearPro':
        variance_scheduler = LinearProScheduler(T=args.T)
    elif args.noise_schedule == 'CosinePro':
        variance_scheduler = CosineProScheduler(T=args.T)
    else:
        raise ValueError("Wrong variance scheduler type !!!")

    # 去噪模型的选择
    if args.denoising_model == 'RDDM_Unet':
        denoising_model = RDDM_Unet(
            dim=args.dim,
            out_dim=args.n_channels,
            channels=args.n_channels,
            input_condition=True,
            input_condition_channels=args.n_channels,
            mask_condition=True,
            mask_condition_channels=args.mask_cls,
            resnet_block_groups=args.resnet_block_groups
        )
    elif args.denoising_model in DiT_models:
        denoising_model = DiT_models[args.denoising_model](
            input_size=args.input_size,
            channels=args.n_channels,
            input_condition=True,
            input_condition_channels=args.n_channels,
            mask_condition=True,
            mask_condition_channels=args.mask_cls,
        )
    else:
        raise ValueError("Wrong denoising_model type !!!")

    resfusion_restore_model = GaussianResfusion_Restore_Mask.load_from_checkpoint(
        checkpoint_path=args.model_ckpt,
        denoising_module=denoising_model,
        variance_scheduler=variance_scheduler,
        mask_cls=args.mask_cls,
        mode=args.mode
    )
    # test the model
    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=True,
        default_root_dir=args.log_dir,
        deterministic='warn',
        precision=args.precision,
        enable_model_summary=False
    )
    trainer.test(model=resfusion_restore_model, datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser('Test the resfusion_restore_mask module')
    # Accuracy control
    parser.add_argument('--set_float32_matmul_precision_high', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_high=False)
    parser.add_argument('--set_float32_matmul_precision_medium', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_medium=False)

    # Basic Test Control
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', default=True, type=bool)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--precision', default='32', type=str)

    # Hyperparameters
    parser.add_argument('--n_channels', default=3, type=int)
    parser.add_argument('--noise_schedule', default='LinearPro', type=str)
    parser.add_argument('--T', default=12, type=int)
    parser.add_argument('--mask_cls', default=2, type=int)

    # Denoising Model Hyperparameters
    parser.add_argument('--denoising_model', default='RDDM_Unet', type=str)
    parser.add_argument('--mode', default='epsilon', type=str)
    parser.add_argument('--model_ckpt', default='', type=str)
    # RDDM_Unet(if used)
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--resnet_block_groups', default=8, type=int)
    # DiT(if used) or DDIM_Unet(if used)
    parser.add_argument('--input_size', default=256, type=int)

    # Test Info
    parser.add_argument('--dataset', default='ISTD', type=str)
    parser.add_argument('--data_dir', default='../datasets/ISTD', type=str)
    parser.add_argument('--log_dir', default='resfusion_restore_mask_test', type=str)

    args = parser.parse_args()

    main(args)
