Please use matlab to run .m files

For the Raindrop dataset, we evaluate PSNR and SSIM based on the luminance channel Y of the YCbCr color space in accordance with previous work.

We conduct experiments on two settings for the Raindrop dataset for fairness, following the methods employed in RDDM and WeatherDiff: 

(1) The results are evaluated at a resolution of $256\times 256$ after being resized. 

`evaluate_PSNR_SSIM_deraining_256.m` 

(2) The original image resolutions are maintained for evaluation.

`evaluate_PSNR_SSIM_deraining_origin.m`