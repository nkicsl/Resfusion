""" Test the resfusion generate module """
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from datamodule import CIFAR10_DataModule
from model.denoising_module import RDDM_Unet, DiT_models, DDIM_Unet
from model import GaussianResfusion_Generate
from variance_scheduler import LinearProScheduler, CosineProScheduler
import torch


def main(args):
    if args.set_float32_matmul_precision_high:
        torch.set_float32_matmul_precision('high')
    if args.set_float32_matmul_precision_medium:
        torch.set_float32_matmul_precision('medium')

    pl.seed_everything(args.seed, workers=True)

    if args.dataset == 'CIFAR10':
        data_module = CIFAR10_DataModule(root_dir=args.data_dir, batch_size=args.batch_size, pin_mem=args.pin_mem,
                                         num_workers=args.num_workers)
    else:
        raise ValueError("Wrong dataset type !!!")

    if args.noise_schedule == 'LinearPro':
        variance_scheduler = LinearProScheduler(T=args.T)
    elif args.noise_schedule == 'CosinePro':
        variance_scheduler = CosineProScheduler(T=args.T)
    else:
        raise ValueError("Wrong variance scheduler type !!!")

    if args.denoising_model == 'RDDM_Unet':
        denoising_model = RDDM_Unet(
            dim=args.dim,
            out_dim=args.n_channels,
            channels=args.n_channels,
            resnet_block_groups=args.resnet_block_groups
        )
    elif args.denoising_model == 'DDIM_Unet':
        denoising_model = DDIM_Unet(
            image_size=args.input_size,
            in_channels=args.n_channels,
            out_ch=args.n_channels
        )
    elif args.denoising_model in DiT_models:
        denoising_model = DiT_models[args.denoising_model](
            input_size=args.input_size,
            channels=args.n_channels
        )
    else:
        raise ValueError("Wrong denoising_model type !!!")

    resfusion_generate_model = GaussianResfusion_Generate.load_from_checkpoint(
        checkpoint_path=args.model_ckpt,
        denoising_module=denoising_model,
        variance_scheduler=variance_scheduler,
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
    trainer.test(model=resfusion_generate_model, datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser('Test the resfusion_generate module')
    # Accuracy control
    parser.add_argument('--set_float32_matmul_precision_high', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_high=False)
    parser.add_argument('--set_float32_matmul_precision_medium', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_medium=False)

    # Basic Test Control
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', default=True, type=bool)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--precision', default='32', type=str)

    # Hyperparameters
    parser.add_argument('--n_channels', default=3, type=int)
    parser.add_argument('--noise_schedule', default='LinearPro', type=str)
    parser.add_argument('--T', default=273, type=int)

    # Denoising Model Hyperparameters
    parser.add_argument('--denoising_model', default='DDIM_Unet', type=str)
    parser.add_argument('--mode', default='epsilon', type=str)
    parser.add_argument('--model_ckpt', default='', type=str)
    # RDDM_Unet(if used)
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--resnet_block_groups', default=8, type=int)
    # DiT(if used) or DDIM_Unet(if used)
    parser.add_argument('--input_size', default=32, type=int)

    # Test Info
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--data_dir', default='../datasets/cifar10', type=str)
    parser.add_argument('--log_dir', default='resfusion_generate_test', type=str)

    args = parser.parse_args()

    main(args)
