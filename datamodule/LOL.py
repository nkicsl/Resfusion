# https://github.com/wyf0912/LLFlow
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize
import numpy as np
import pytorch_lightning as pl

# def visualize(image, mask, original_image=None, original_gt=None):
#     fontsize = 18
#
#     if original_image is None and original_gt is None:
#         f, ax = plt.subplots(2, 1, figsize=(8, 8))
#         ax[0].imshow(image)
#         ax[1].imshow(mask)
#     else:
#         f, ax = plt.subplots(2, 2, figsize=(8, 8))
#
#         ax[0, 0].imshow(original_image)
#         ax[0, 0].set_title('Original image', fontsize=fontsize)
#
#         ax[1, 0].imshow(original_gt)
#         ax[1, 0].set_title('Original gt', fontsize=fontsize)
#
#         ax[0, 1].imshow(image)
#         ax[0, 1].set_title('Transformed image', fontsize=fontsize)
#
#         ax[1, 1].imshow(mask)
#         ax[1, 1].set_title('Transformed gt', fontsize=fontsize)
#
#
# my_transform_train = A.Compose([
#     A.RandomCrop(height=256, width=256),
#     A.HorizontalFlip(p=0.5)],
#     additional_targets={'gt_image': 'image'},
# )
#
# my_transform_test = A.Compose([
#     additional_targets={'gt_image': 'image'},
# )

##################################################################
train_transform = A.Compose([
    A.RandomCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2()],
    additional_targets={'gt_image': 'image'},
)

test_transform = A.Compose([
    Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2()],
    additional_targets={'gt_image': 'image'},
)
##################################################################


class LOLDataset(Dataset):
    def __init__(self, root_dir, subset="our485", transform=None):
        """
        初始化函数
        :param root_dir: 数据集的根目录
        :param subset: 使用的数据子集 ('our485', 'eval15')
        :param transform: 应用于数据的转换
        """
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.img_dir = os.path.join(root_dir, subset, "low")
        self.gt_dir = os.path.join(root_dir, subset, "high")
        self.img_names = os.listdir(self.img_dir)
        self.img_names.sort()
        if subset == 'our485':
            assert len(self.img_names) == 485
        else:
            assert len(self.img_names) == 15

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)
        # 读取图像
        image = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        # 将PIL图像转换为NumPy数组
        image = np.array(image)
        gt = np.array(gt)
        if self.transform:
            augmented = self.transform(image=image, gt_image=gt)
            image = augmented['image']
            gt = augmented['gt_image']
        img_input = image
        target = gt
        return img_input, target


class LOLDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str = "../datasets/LOLdataset", batch_size=1, pin_mem=True, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.pin_mem = pin_mem
        self.num_workers = num_workers

    '''
    setup 方法创建Dataset对象，对不同数据集指定预处理方法
    '''

    def setup(self, stage: str):
        # Assign train/val dataloader for use in dataloaders
        if stage == "fit":
            self.train_dataset = LOLDataset(root_dir=self.root_dir, subset="our485", transform=train_transform)
            self.val_dataset = LOLDataset(root_dir=self.root_dir, subset="eval15", transform=test_transform)
        # Assign test dataset for use in dataloader(s)
        if stage == "validate":
            self.val_dataset = LOLDataset(root_dir=self.root_dir, subset="eval15", transform=test_transform)
        if stage == "test":
            self.test_dataset = LOLDataset(root_dir=self.root_dir, subset="eval15", transform=test_transform)

    # 以下方法创建不同阶段的数据加载器
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, drop_last=True,
                                           batch_size=self.batch_size, pin_memory=self.pin_mem,
                                           num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, drop_last=False,
                                           batch_size=self.batch_size, pin_memory=self.pin_mem,
                                           num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, shuffle=False, drop_last=False,
                                           batch_size=self.batch_size, pin_memory=self.pin_mem,
                                           num_workers=self.num_workers, persistent_workers=True)

# # test code
# if __name__ == '__main__':
#     my_dataset = LOLDataset(root_dir='../../datasets/LOLdataset')
#     input, target = my_dataset.__getitem__(100)
#
#     augmented = my_transform_test(image=input, gt_image=target)
#     input_medium = augmented['image']
#     target_medium = augmented['gt_image']
#
#     augmented = test_transform(image=input, gt_image=target)
#     input_medium2 = augmented['image']
#     target_medium2 = augmented['gt_image']
#
#     # 展示图像和掩码
#     visualize(input_medium, target_medium, original_image=input, original_gt=target)
#     plt.show()
#
#     from torchmetrics.image import PeakSignalNoiseRatio
#     from skimage import metrics
#
#     skimage_pnsr = metrics.peak_signal_noise_ratio(input_medium,
#                                                    target_medium,
#                                                    data_range=255)
#     print('skimage_pnsr', skimage_pnsr)
#     psnr = PeakSignalNoiseRatio(data_range=(0, 1))
#     torchmetrics_pnsr = psnr(preds=input_medium2, target=target_medium2)
#     print('torchmetrics_pnsr', torchmetrics_pnsr)
#     print(abs(skimage_pnsr - torchmetrics_pnsr) < 0.001)
#
#     from torchmetrics.image import StructuralSimilarityIndexMeasure
#
#     ssim = StructuralSimilarityIndexMeasure(data_range=(0, 1))
#     torchmetrics_ssim = ssim(preds=input_medium2.unsqueeze(0), target=target_medium2.unsqueeze(0))
#     print('torchmetrics_ssim', torchmetrics_ssim)
