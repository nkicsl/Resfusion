from typing import Callable, Iterator, Tuple, Optional, Type, Union, List, ClassVar

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F

from .distributions import resfusion_x0_to_xt, ddpm_x0_to_xt
from variance_scheduler.abs_var_scheduler import Scheduler
from torch import optim
from torchmetrics.image.psnr import PeakSignalNoiseRatio


class GaussianResfusion_Restore(pl.LightningModule):
    """
    Gaussian Prior Residual Noise embedded De-noising Diffusion Probabilistic Model
    """

    def __init__(self, denoising_module: pl.LightningModule, variance_scheduler: Scheduler,
                 mode='epsilon',
                 loss_type='L2', optimizer_type='Adam', lr_scheduler_type='CosineAnnealingLR', **kwargs):
        """
        :param denoising_module: the nn which computes the denoise step i.e. q(x_{t-1} | x_t, x_0, R)
        :param variance_scheduler: the variance scheduler cited in DDPM paper. See folder variance_scheduler
        :param mode: 'epsilon' or 'sample' or 'residual'
        """
        super().__init__()
        # lightning code
        self.save_hyperparameters(ignore=['denoising_module', 'variance_scheduler'])
        # sample mode
        self.mode = mode
        # loss type
        self.loss_type = loss_type
        # optimizer type
        self.optimizer_type = optimizer_type
        # lr_scheduler type
        self.lr_scheduler_type = lr_scheduler_type

        # initialize models
        self.denoising_module = denoising_module

        # PSNR
        self.val_PSNR = PeakSignalNoiseRatio(dim=(1, 2, 3), data_range=(0, 1))
        self.test_PSNR = PeakSignalNoiseRatio(dim=(1, 2, 3), data_range=(0, 1))

        # get the variance scheduler cited in DDPM paper
        self.var_scheduler = variance_scheduler
        self.alphas_hat = self.var_scheduler.get_alphas_hat()
        self.alphas = self.var_scheduler.get_alphas()
        self.betas = self.var_scheduler.get_betas()
        self.betas_hat = self.var_scheduler.get_betas_hat()
        self.alphas_hat_t_minus_1 = self.var_scheduler.get_alphas_hat_t_minus_1()
        # acc point
        self.T_acc = self.get_acc_point(self.alphas_hat).item()
        print('acc point:', self.T_acc)

    def get_acc_point(self, alphas_hat):
        """
            Calculate Acceleration Point, according to formula 27
        """
        abs_dist = torch.abs(torch.sqrt(alphas_hat) - 0.5)
        return abs_dist.argmin() + 1

    def forward(self, x_t: torch.FloatTensor, I_in: torch.FloatTensor, t: int) -> torch.Tensor:
        """
        Forward pass of the Resfusion model.

        Args:
            x_t: noised Input image tensor.
            I_in: condition image tensor.
            t: Time step tensor.

        Returns:
            predicted noise tensor or predicted x_0 tensor.
        """
        return self.denoising_module(x_t, I_in, t)

    def training_step(self, batch, batch_idx: int):
        """
        Training step of the Resfusion model.

        Args:
            batch:
                input: normalized degraded picture, [0, 1]
                target: normalized ground truth picture, [0, 1]
            batch_idx: Batch index.

        Returns:
            loss.
        """
        inputs, targets = batch
        # initialize X_0 and X_0_hat
        X_0 = targets
        X_0_hat = inputs
        # Map image values from [0, 1] to [-1, 1]
        X_0 = X_0 * 2 - 1
        X_0_hat = X_0_hat * 2 - 1
        # Compute the residual term following formula 16
        residual_term = X_0_hat - X_0
        # Sample a random time step t from 0 to T_acc-1 for each image in the batch
        # Uniform Sampling
        t: torch.Tensor = torch.randint(0, self.T_acc, (X_0.shape[0],), device=self.device)
        # Compute alpha_hat for the selected time steps
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1)
        # Sample noise from a normal distribution with the same shape as X_0
        noise = torch.randn_like(X_0)
        # Compute the intermediate image x_t from the original image X_0, alpha_hat, residual term and noise
        x_t = resfusion_x0_to_xt(X_0, alpha_hat, residual_term, noise)  # go from x_0 to x_t with the formula 20
        if self.mode == 'epsilon':
            # Compute alpha, beta for the selected time steps
            alpha = self.alphas[t].reshape(-1, 1, 1, 1)
            beta = self.betas[t].reshape(-1, 1, 1, 1)
            # Compute resnoise with formula 24
            resnoise = noise + (1 - torch.sqrt(alpha)) * torch.sqrt(1 - alpha_hat) / beta * residual_term
            # Run the intermediate image x_t through the model to obtain predicted resnoise
            pred_resnoise = self.denoising_module(x=x_t, time=t, input_cond=X_0_hat)
            # Loss
            if self.loss_type == 'L2':
                # Compute the MSE loss for the predicted noise
                loss = F.mse_loss(input=pred_resnoise, target=resnoise)
            elif self.loss_type == 'L1':
                # Compute the smooth L1 loss for the predicted noise
                loss = F.smooth_l1_loss(input=pred_resnoise, target=resnoise)
            else:
                raise ValueError("Wrong loss type !!!")
        elif self.mode == 'sample':
            # Run the intermediate image x_t through the model to obtain predicted x_0
            pred_x_0 = self.denoising_module(x=x_t, time=t, input_cond=X_0_hat)
            # Loss
            if self.loss_type == 'L2':
                # Compute the MSE loss for the predicted x_0
                loss = F.mse_loss(input=pred_x_0, target=X_0)
            elif self.loss_type == 'L1':
                # Compute the smooth L1 loss for the predicted noise
                loss = F.smooth_l1_loss(input=pred_x_0, target=X_0)
            else:
                raise ValueError("Wrong loss type !!!")
        elif self.mode == 'residual':
            # Run the intermediate image x_t through the model to obtain predicted residual_term
            pred_residual_term = self.denoising_module(x=x_t, time=t, input_cond=X_0_hat)
            # Loss
            if self.loss_type == 'L2':
                # Compute the MSE loss for the predicted x_0
                loss = F.mse_loss(input=pred_residual_term, target=residual_term)
            elif self.loss_type == 'L1':
                # Compute the smooth L1 loss for the predicted noise
                loss = F.smooth_l1_loss(input=pred_residual_term, target=residual_term)
        else:
            raise ValueError("Wrong mode !!!")
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch
        X_0 = targets
        X_0_hat = inputs
        # Map image values from [0, 1] to [-1, 1]
        X_0_hat = X_0_hat * 2 - 1
        pred_x_0 = self.generate(X_0_hat)
        # threshold clip
        pred_x_0 = torch.clamp(pred_x_0, min=-1, max=1)
        # rescale from [-1, 1] to [0, 1]
        pred_x_0 = (pred_x_0 + 1) / 2
        # calculate PSNR
        self.val_PSNR(preds=pred_x_0, target=X_0)
        self.log('val_PSNR', self.val_PSNR, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # log images
        if batch_idx < 4:
            self.log_tb_images((inputs, targets, pred_x_0), batch_idx, self.current_epoch, save_all=False)

    def test_step(self, batch, batch_idx: int):
        inputs, targets = batch
        X_0 = targets
        X_0_hat = inputs
        # Map image values from [0, 1] to [-1, 1]
        X_0_hat = X_0_hat * 2 - 1
        pred_x_0 = self.generate(X_0_hat)
        # threshold clip
        pred_x_0 = torch.clamp(pred_x_0, min=-1, max=1)
        # rescale from [-1, 1] to [0, 1]
        pred_x_0 = (pred_x_0 + 1) / 2
        # calculate PSNR
        self.test_PSNR(preds=pred_x_0, target=X_0)
        self.log('test_PSNR', self.test_PSNR, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # log images
        self.log_tb_images((inputs, targets, pred_x_0), batch_idx, self.current_epoch, save_all=True)

    def configure_optimizers(self):
        blr = self.hparams.blr
        eff_batch_size = self.hparams.batch_size * self.hparams.accum_iter * self.hparams.devices * self.hparams.num_nodes
        lr = blr * eff_batch_size / 256

        if self.optimizer_type == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                                   weight_decay=self.hparams.weight_decay, amsgrad=True)
        elif self.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                                    weight_decay=self.hparams.weight_decay, amsgrad=True)
        elif self.optimizer_type == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                                  weight_decay=self.hparams.weight_decay, momentum=0.9, nesterov=True)
        else:
            raise ValueError("Wrong optimizer type !!!")

        if self.lr_scheduler_type == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs,
                                                             eta_min=self.hparams.min_lr)
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler}
        elif self.lr_scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20,
                                                             cooldown=20, min_lr=self.hparams.min_lr)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler,
                                     "monitor": "train_loss_epoch"}}
        else:
            raise ValueError("Wrong lr scheduler type !!!")

    def generate(self, X_0_hat: torch.Tensor,
                 get_intermediate_steps: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate a batch of images via resfusion
        :param X_0_hat: the scaled inputs
        :param get_intermediate_steps: return all the denoising steps instead of the final step output
        :return: The tensor [bs, c, w, h] of generated images or a list of tensors [bs, c, w, h] if get_intermediate_steps=True
        """
        if get_intermediate_steps:
            steps = []
        # generate x_T_acc with formula 28
        # little modification since sqrt(alpha_hat)=0.5, this reduces the difference
        alpha_hat = self.alphas_hat[self.T_acc - 1]
        noise = torch.randn_like(X_0_hat)
        X_noise = ddpm_x0_to_xt(X_0_hat, alpha_hat, noise)
        for t in range(self.T_acc - 1, -1, -1):
            alpha_t = self.alphas[t]
            alpha_hat_t = self.alphas_hat[t]
            beta_t = self.betas[t]
            beta_hat_t = self.betas_hat[t]
            alpha_hat_t_minus_1 = self.alphas_hat_t_minus_1[t]
            if get_intermediate_steps:
                steps.append(X_noise)
            t_tensor = torch.LongTensor([t]).to(self.device).expand(X_noise.shape[0])
            # the variance is taken fixed as in the original DDPM paper
            sigma = torch.sqrt(beta_hat_t)
            z = torch.randn_like(X_noise)
            if self.mode == 'epsilon':
                pred_resnoise = self.denoising_module(x=X_noise, time=t_tensor,
                                                      input_cond=X_0_hat)  # predict the resnoise
                if t == 0:
                    z.fill_(0)
                # denoise step from x_t to x_{t-1} with formula 33
                X_noise = 1 / (torch.sqrt(alpha_t)) * \
                          (X_noise - (beta_t / torch.sqrt(1 - alpha_hat_t)) * pred_resnoise) + sigma * z
            elif self.mode == 'sample':
                pred_x_0 = self.denoising_module(x=X_noise, time=t_tensor, input_cond=X_0_hat)  # predict the x_0
                # threshold clip
                pred_x_0 = torch.clamp(pred_x_0, min=-1, max=1)
                if t == 0:
                    X_noise = pred_x_0
                else:
                    pred_residual_term = X_0_hat - pred_x_0
                    # denoise step from x_t to x_{t-1} with formula 44
                    X_noise = (((torch.sqrt(alpha_t) * (1 - alpha_hat_t_minus_1)) * (X_noise - pred_residual_term)
                                + (torch.sqrt(alpha_hat_t_minus_1)) * (1 - alpha_t) * (pred_x_0 - pred_residual_term))
                               / (1 - alpha_hat_t)
                               + pred_residual_term + sigma * z)
            elif self.mode == 'residual':
                pred_residual_term = self.denoising_module(x=X_noise, time=t_tensor,
                                                           input_cond=X_0_hat)  # predict the x_0
                pred_x_0 = X_0_hat - pred_residual_term
                # threshold clip
                pred_x_0 = torch.clamp(pred_x_0, min=-1, max=1)
                if t == 0:
                    X_noise = pred_x_0
                else:
                    # threshold clip
                    pred_residual_term = X_0_hat - pred_x_0
                    # denoise step from x_t to x_{t-1} with formula 44
                    X_noise = (((torch.sqrt(alpha_t) * (1 - alpha_hat_t_minus_1)) * (X_noise - pred_residual_term)
                                + (torch.sqrt(alpha_hat_t_minus_1)) * (1 - alpha_t) * (pred_x_0 - pred_residual_term))
                               / (1 - alpha_hat_t)
                               + pred_residual_term + sigma * z)
            else:
                raise ValueError("Wrong mode !!!")
        if get_intermediate_steps:
            steps.append(X_noise)
            return steps
        return X_noise

    def on_fit_start(self) -> None:
        self.alphas_hat = self.alphas_hat.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.betas = self.betas.to(self.device)
        self.betas_hat = self.betas_hat.to(self.device)
        self.alphas_hat_t_minus_1 = self.alphas_hat_t_minus_1.to(self.device)

    def on_test_start(self) -> None:
        self.alphas_hat = self.alphas_hat.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.betas = self.betas.to(self.device)
        self.betas_hat = self.betas_hat.to(self.device)
        self.alphas_hat_t_minus_1 = self.alphas_hat_t_minus_1.to(self.device)

    def log_tb_images(self, viz_batch, batch_idx, current_epoch, save_all=False) -> None:

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        # Log the images (Give them different names)
        for img_idx, (image, y_true, y_pred) in enumerate(zip(*viz_batch)):
            if save_all:
                tb_logger.add_image("Image/{:04d}_{:04d}".format(batch_idx, img_idx), image, 0)
                tb_logger.add_image("GroundTruth/{:04d}_{:04d}".format(batch_idx, img_idx), y_true, 0)
                tb_logger.add_image("Prediction/{:04d}_{:04d}".format(batch_idx, img_idx), y_pred, current_epoch)
            else:
                if img_idx < 8:
                    tb_logger.add_image(f"Image/{batch_idx}_{img_idx}", image, 0)
                    tb_logger.add_image(f"GroundTruth/{batch_idx}_{img_idx}", y_true, 0)
                    tb_logger.add_image(f"Prediction/{batch_idx}_{img_idx}", y_pred, current_epoch)
