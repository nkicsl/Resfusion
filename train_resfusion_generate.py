""" Train the resfusion generate module """
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from datamodule import CIFAR10_DataModule
from model.denoising_module import RDDM_Unet, DiT_models, DDIM_Unet
from model import GaussianResfusion_Generate
from variance_scheduler import LinearProScheduler, CosineProScheduler
from callback import EMA, EMAModelCheckpoint
import torch


def load_callbacks(args):
    callbacks = []

    if args.use_ema:
        callbacks.append(EMAModelCheckpoint(
            monitor='val_FID',
            filename='best-{epoch:02d}-{val_FID:.3f}',
            mode='min',
            save_last=True,
            save_on_train_epoch_end=True,
            every_n_epochs=args.check_val_every_n_epoch
        ))
        callbacks.append(EMA(decay=0.9999))
    else:
        callbacks.append(plc.ModelCheckpoint(
            monitor='val_FID',
            filename='best-{epoch:02d}-{val_FID:.3f}',
            mode='min',
            save_last=True,
            save_on_train_epoch_end=True,
            every_n_epochs=args.check_val_every_n_epoch
        ))

    callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))

    if args.early_stopping:
        callbacks.append(plc.EarlyStopping(monitor='val_FID', mode='min', patience=50))

    return callbacks


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

    resfusion_generate_model = GaussianResfusion_Generate(denoising_module=denoising_model,
                                                          variance_scheduler=variance_scheduler,
                                                          **vars(args))
    # train the model
    trainer = Trainer(
        log_every_n_steps=1,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accum_iter,
        default_root_dir=args.log_dir,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip,
        precision=args.precision,
        logger=True,
        callbacks=load_callbacks(args),
        deterministic='warn',
        strategy='ddp',
        enable_model_summary=False
    )
    trainer.fit(model=resfusion_generate_model, datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser('Train the resfusion_generate module')
    # Accuracy control
    parser.add_argument('--set_float32_matmul_precision_high', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_high=False)
    parser.add_argument('--set_float32_matmul_precision_medium', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_medium=False)

    # Basic Training Control
    parser.add_argument('--epochs', default=3000, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations '
                             '(for increasing the effective batch size under memory constraints)')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', default=True, type=bool)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--gradient_clip', default=1, type=float)
    parser.add_argument('--precision', default='32', type=str)
    parser.add_argument('--early_stopping', action='store_true')
    parser.set_defaults(early_stopping=False)
    parser.add_argument('--use_ema', action='store_true')
    parser.set_defaults(use_ema=False)

    # Hyperparameters
    parser.add_argument('--n_channels', default=3, type=int)
    parser.add_argument('--noise_schedule', default='LinearPro', type=str)
    parser.add_argument('--T', default=273, type=int)
    parser.add_argument('--loss_type', default='L2', type=str)
    parser.add_argument('--optimizer_type', default='AdamW', type=str)
    parser.add_argument('--lr_scheduler_type', default='CosineAnnealingLR', type=str)

    # Denoising Model Hyperparameters
    parser.add_argument('--denoising_model', default='DDIM_Unet', type=str)
    parser.add_argument('--mode', default='epsilon', type=str)
    # RDDM_Unet(if used)
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--resnet_block_groups', default=8, type=int)
    # DiT(if used) or DDIM_Unet(if used)
    parser.add_argument('--input_size', default=32, type=int)

    # Optimizer parameters
    parser.add_argument('--blr', default=4e-4, type=float)
    parser.add_argument('--min_lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)

    # Training Info
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--data_dir', default='../datasets/cifar10', type=str)
    parser.add_argument('--log_dir', default='resfusion_generate_train', type=str)

    # distributed training parameters
    parser.add_argument('--accelerator', default="gpu", type=str,
                        help='type of accelerator')
    parser.add_argument('--devices', default=2, type=int,
                        help='number of devices')
    parser.add_argument('--num_nodes', default=1, type=int,
                        help='number of num nodes')

    args = parser.parse_args()

    main(args)
