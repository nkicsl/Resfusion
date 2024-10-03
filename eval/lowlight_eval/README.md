Please use matlab to run .m files. You may need to `pip install lpips` before running `.py` files.

We conducted experiments on two settings for the LOL dataset, following the methods employed in RDDM and LLFormer: 

(1) The results are evaluated at a resolution of $256\times 256$ after being resized. PSNR and SSIM are evaluated based in YCbCr color space. 

`Measure_256_lpips.py` for PSNR, SSIM, LPIPS at a resolution of $256\times 256$ in RGB color space.

`evaluate_PSNR_SSIM_256.m` for PSNR, SSIM at a resolution of $256\times 256$ in YCbCr color space.

(2) The original image resolutions ($600\times 400$) are maintained for evaluation. PSNR and SSIM are evaluated in RGB color space.

`Measure.py`