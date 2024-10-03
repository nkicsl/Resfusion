# Resfusion: Denoising Diffusion Probabilistic Models for Image Restoration Based on Prior Residual Noise

![image](https://github.com/nkicsl/Resfusion/blob/main/assets/Visualization.jpg)

arxiv link: https://arxiv.org/abs/2311.14900

This repository is the official [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) implementation for Resfusion: Denoising Diffusion Probabilistic Models for Image Restoration Based on Prior Residual Noise.

## Directory Structure
1. `callback` mainly used to store the implementation of EMA as a callback.
2. `datamodule` mainly used for storing datamodules for different datasets.
3. `eval` mainly stores code for evaluation.
4. `model` mainly stores the main Resfusion models and its denoising backbones.
5. `.py` starting with `train` are for training, while those starting with `test` are for testing.

## Environment Setup
    conda env create -f environment.yaml

## Dataset Download

[ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN)

[LOL](https://daooshee.github.io/BMVC2018website/)

[Raindrop](https://github.com/rui1996/DeRaindrop)

Please download them to the `datasets` directory and organize them as follows:
```
├── resfusion-master
├── datasets
    ├── ISTD
        ├── train
        ├── test
    ├── LOLdataset
        ├── our485
        ├── eval15
    ├── Raindrop
        ├── train
        ├── test_a
```

## Training Pipeline

ISTD Dataset
    
    python train_resfusion_restore_mask.py --num_workers 24 --T 12 --batch_size 4 --device 8 --denoising_model RDDM_Unet 

LOL Dataset
    
    python train_resfusion_restore.py --num_workers 24 --T 12 --denoising_model RDDM_Unet

Raindrop Dataset
    
    python train_resfusion_restore.py --T 12 --dataset Raindrop --data_dir ../datasets/Raindrop --batch_size 4 --device 8

CIFAR10 Dataset (example with 100 sampling steps)
    
    python train_resfusion_generate.py --T 273 --num_workers 24 --batch_size 128 --devices 1 --blr 4e-4 --min_lr 2e-4 --use_ema


## Testing Pipeline

Step 1: Run the testing script (taking ISTD dataset as an example)

ISTD dataset

    python test_resfusion_restore_mask.py --T 12 --model_ckpt ./ckpt/ISTD/best-epoch\=2639-val_PSNR\=30.068.ckpt --seed 42

Step 2: Export generated prediction images using `./eval/save_images_for_test.ipynb`

Step 3: Align the names of exported prediction images with real test dataset images using `./eval/name_alignment.ipynb`

Step 4: Assess quantitative metrics using MATLAB files and .py files in `./eval`

## Results Download
| Dataset          | results                                                                                                                |
|------------------|------------------------------------------------------------------------------------------------------------------------|
| ISTD dataset     | [Resfusion_ISTD.zip](https://1drv.ms/u/c/e619d8638269023a/EY2RmFGbjk1ApWQR7m2mPjYB1bmikLfadagBNBlm65aSRg?e=t9m1cV)     |
| LOL dataset      | [Resfusion_LOL.zip](https://1drv.ms/u/c/e619d8638269023a/EQRpiq5ur5dHkVfPP4vK-KkBHO9rDvl-4ch8PnvdPrE8SA?e=Coi0LY)      |
| Raindrop dataset | [Resfusion_Raindrop.zip](https://1drv.ms/u/c/e619d8638269023a/EY4X1OIdGMtIqyMmwDRIrrIBrb7uHSFG5Ql6LrzAOlpl2A?e=WcE971) |

## Estimation of Parameters and MACs
Consistent with [RDDM](https://github.com/nachifur/RDDM), we used [THOP](https://github.com/Lyken17/pytorch-OpCounter) to assess the parameters and MACs, see the code in `./eval/cal_params_and_macs.py`

## Truncated Strategy
We have provided a mapping table `./assets/acc_T_change_table.xlsx` between $T'$ and $T$ in our `truncated schedule`, along with the corresponding curve graph for $\sqrt{\overline{\alpha}_{t}}$.

## Tips
1. Strictly adhere to the hyperparameters set during training when testing the model.

## Thanks
Thanks to [MulimgViewer](https://github.com/nachifur/MulimgViewer) for the support in generating visual comparison results, and special thanks to [@ObscureLin](https://github.com/ObscureLin) for the technique support throughout the project.

## Citation
If you find this work useful for your research, please consider citing:
```
@article{zhenning2023resfusion,
  title={Resfusion: Prior Residual Noise embedded Denoising Diffusion Probabilistic Models},
  author={Zhenning, Shi and Changsheng, Dong and Bin, Pan and Xueshuo, Xie and Along, He and Qiaoying, Qu and Tao, Li},
  journal={arXiv preprint arXiv:2311.14900},
  year={2023}
}
```