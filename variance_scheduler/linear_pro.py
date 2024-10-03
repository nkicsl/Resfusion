# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
import torch
import numpy as np

from variance_scheduler.abs_var_scheduler import Scheduler


# safer and equal codes, can be used in the future
# def find_closest_index(lst: torch.Tensor, target):
#     for i in range(len(lst)):
#         if lst[i] <= 0.5:
#             return i
def find_closest_index(lst: torch.Tensor, target):
    lo, hi = 0, lst.size(0) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if lst[mid] <= target:
            hi = mid - 1
        else:
            lo = mid + 1
    return lo


class LinearProScheduler(Scheduler):
    """
        A pro scheduler that linearly interpolates between two values of beta over T steps.
        """

    def __init__(self, T: int):
        """
        Initializes the LinearProScheduler.

        Args:
            T (int): The number of steps to interpolate over.
        """
        # Linear schedule from Ho et al, extended to work for any number of diffusion steps.
        self.T = T
        self.scale = 1000 / T
        self.beta_start = self.scale * 0.0001
        self.beta_end = self.scale * 0.02
        self._beta = torch.linspace(self.beta_start, self.beta_end, T)
        self._alpha = 1.0 - self._beta
        self._alpha_hat = torch.cumprod(self._alpha, dim=0)

        # Add an offset to make sure that there is a certain acceleration point equal to 0.5 in _sqrt_alpha_hat
        # abs_dist = torch.abs(self._sqrt_alpha_hat - 0.5)
        # min_idx = abs_dist.argmin().int()
        # offset = 0.5 - self._sqrt_alpha_hat[min_idx]
        # self._sqrt_alpha_hat = self._sqrt_alpha_hat + offset
        # self._alpha_hat = torch.pow(self._sqrt_alpha_hat, 2)
        # for i in range(1, len(self._alpha)):
        #     self._alpha[i] = self._alpha_hat[i] / self._alpha_hat[i - 1]
        # self._beta = 1.0 - self._alpha

        # 找到sqrt_alpha_hat第一个比0.5小的，如果差大于0.01则把它变为0.5并截断
        self._sqrt_alpha_hat = torch.sqrt(self._alpha_hat)
        idx = find_closest_index(self._sqrt_alpha_hat, 0.5)
        if 0.5 - self._sqrt_alpha_hat[idx].item() > 0.01:
            # The value after the acceleration point is useless and can be discarded
            self._sqrt_alpha_hat = torch.cat((self._sqrt_alpha_hat[:idx], torch.tensor([0.5])))
            self._alpha_hat = torch.cat((self._alpha_hat[:idx], torch.tensor([0.5 ** 2])))
            self._alpha = torch.cat((self._alpha[:idx], torch.tensor([0.5 ** 2]) / self._alpha_hat[idx-1]))
            self._beta = 1.0 - self._alpha

        self._alpha_hat_t_minus_1 = torch.roll(self._alpha_hat, shifts=1, dims=0)
        self._alpha_hat_t_minus_1[0] = self._alpha_hat_t_minus_1[1]
        self._beta_hat = (1 - self._alpha_hat_t_minus_1) / (1 - self._alpha_hat) * self._beta

    def get_alphas_hat(self):
        """
        Returns the cumulative product of (1 - beta) over time.
        """
        return self._alpha_hat

    def get_alphas(self):
        """
        Returns the values of alpha over time.
        """
        return self._alpha

    def get_betas(self):
        """
        Returns the values of beta over time.
        """
        return self._beta

    def get_betas_hat(self):
        """
        Returns the values of beta_hat over time.
        """
        return self._beta_hat

    def get_alphas_hat_t_minus_1(self):
        """
        Returns the values of alpha_hat_t_minus_1 over time.
        """
        return self._alpha_hat_t_minus_1


if __name__ == '__main__':
    scheduler = LinearProScheduler(20)

    beta = scheduler.get_betas()
    alpha = scheduler.get_alphas()
    alpha_hat = scheduler.get_alphas_hat()
    sqrt_alpha_hat = torch.sqrt(alpha_hat)

    import matplotlib.pyplot as plt
    plt.plot(sqrt_alpha_hat.numpy())
    plt.ylabel('$\\alpha_t$')
    plt.xlabel('t')
    plt.show()
    print(len(sqrt_alpha_hat))


