import torch


def resfusion_x0_to_xt(x_0: torch.Tensor, alpha_hat_t: torch.Tensor, residual_term: torch.Tensor, noise: torch.Tensor) \
        -> torch.Tensor:
    """
    Compute x_t from x_0 with formula 20 from Resfusion
    :param x_0: the image without noise
    :param alpha_hat_t: the cumulated variance schedule at time t
    :param residual_term: the residual term
    :param noise: pure noise from N(0, 1)
    :return: the noised image x_t at step t
    """
    return (torch.sqrt(alpha_hat_t) * x_0 + (1 - torch.sqrt(alpha_hat_t)) * residual_term
            + torch.sqrt(1 - alpha_hat_t) * noise)


def ddpm_x0_to_xt(x_0: torch.Tensor, alpha_hat_t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Compute x_t from x_0 using a closed form using theorem from original DDPM paper (Ho et al.)
    :param x_0: the image without noise
    :param alpha_hat_t: the cumulated variance schedule at time t
    :param noise: pure noise from N(0, 1)
    :return: the noised image x_t at step t
    """
    return torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * noise
