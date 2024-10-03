# following https://github.com/Lyken17/pytorch-OpCounter/blob/master/README.md
import torch
from model.denoising_module import RDDM_Unet
from thop import profile
from thop import clever_format

denoising_model = RDDM_Unet(
    dim=64,
    out_dim=3,
    channels=3,
    input_condition=True,
    input_condition_channels=3,
    resnet_block_groups=8
).cuda()

x_t = torch.randn(1, 3, 256, 256).cuda()
time = torch.randint(0, 10, (1,)).cuda()
input_cond = torch.randn(1, 3, 256, 256).cuda()
macs, params = profile(denoising_model, inputs=(x_t, time, input_cond))
macs, params = clever_format([macs, params], "%.3f")
print('no mask cond')
print('macs: ', macs, 'params: ', params)

denoising_model = RDDM_Unet(
    dim=64,
    out_dim=3,
    channels=3,
    input_condition=True,
    input_condition_channels=3,
    mask_condition=True,
    mask_condition_channels=2,
    resnet_block_groups=8
).cuda()
mask_cond = torch.randn(1, 2, 256, 256).cuda()
macs, params = profile(denoising_model, inputs=(x_t, time, input_cond, mask_cond))
macs, params = clever_format([macs, params], "%.3f")
print('mask cond')
print('macs: ', macs, 'params: ', params)
