""" Train the resfusion restore module """
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from datamodule import ISTD_DataModule
from model.denoising_module import RDDM_Unet, DiT_models
from model import GaussianResfusion_Restore_Mask
from variance_scheduler import LinearProScheduler, CosineProScheduler
import torch


def load_callbacks(args):
    callbacks = []

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_PSNR',
        filename='best-{epoch:02d}-{val_PSNR:.3f}',
        mode='max',
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=args.check_val_every_n_epoch
    ))

    callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))

    if args.early_stopping:
        callbacks.append(plc.EarlyStopping(monitor='val_PSNR', mode='max', patience=50))

    return callbacks


def main(args):
    if args.set_float32_matmul_precision_high:
        torch.set_float32_matmul_precision('high')
    if args.set_float32_matmul_precision_medium:
        torch.set_float32_matmul_precision('medium')

    pl.seed_everything(args.seed, workers=True)

    if args.dataset == 'ISTD':
        data_module = ISTD_DataModule(root_dir=args.data_dir, batch_size=args.batch_size, pin_mem=args.pin_mem,
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

    resfusion_restore_model = GaussianResfusion_Restore_Mask(denoising_module=denoising_model,
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
    trainer.fit(model=resfusion_restore_model, datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser('Train the resfusion_restore_mask module')
    # Accuracy control
    parser.add_argument('--set_float32_matmul_precision_high', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_high=False)
    parser.add_argument('--set_float32_matmul_precision_medium', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_medium=False)

    # Basic Training Control
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations '
                             '(for increasing the effective batch size under memory constraints)')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', default=True, type=bool)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--gradient_clip', default=1, type=float)
    parser.add_argument('--precision', default='32', type=str)
    parser.add_argument('--early_stopping', action='store_true')
    parser.set_defaults(early_stopping=False)

    # Hyperparameters
    parser.add_argument('--n_channels', default=3, type=int)
    parser.add_argument('--mask_cls', default=2, type=int)
    parser.add_argument('--noise_schedule', default='LinearPro', type=str)
    parser.add_argument('--T', default=12, type=int)
    parser.add_argument('--loss_type', default='L2', type=str)
    parser.add_argument('--optimizer_type', default='AdamW', type=str)
    parser.add_argument('--lr_scheduler_type', default='CosineAnnealingLR', type=str)

    # Denoising Model Hyperparameters
    parser.add_argument('--denoising_model', default='RDDM_Unet', type=str)
    parser.add_argument('--mode', default='epsilon', type=str)
    # RDDM_Unet(if used)
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--resnet_block_groups', default=8, type=int)
    # DiT(if used) or DDIM_Unet(if used)
    parser.add_argument('--input_size', default=256, type=int)

    # Optimizer parameters
    parser.add_argument('--blr', default=8.8e-4, type=float)
    parser.add_argument('--min_lr', default=3e-5, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)

    # Training Info
    parser.add_argument('--dataset', default='ISTD', type=str)
    parser.add_argument('--data_dir', default='../datasets/ISTD', type=str)
    parser.add_argument('--log_dir', default='resfusion_restore_mask_train', type=str)

    # distributed training parameters
    parser.add_argument('--accelerator', default="gpu", type=str,
                        help='type of accelerator')
    parser.add_argument('--devices', default=2, type=int,
                        help='number of devices')
    parser.add_argument('--num_nodes', default=1, type=int,
                        help='number of num nodes')

    args = parser.parse_args()

    main(args)
