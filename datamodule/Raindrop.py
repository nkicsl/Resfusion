# https://github.com/IGITUGraz/WeatherDiffusion
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


class RaindropDataset(Dataset):
    def __init__(self, root_dir, subset="train", transform=None):
        """
        初始化函数
        :param root_dir: 数据集的根目录
        :param subset: 使用的数据子集 ('train', 'test_a')
        :param transform: 应用于数据的转换
        """
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.img_dir = os.path.join(root_dir, subset, "data")
        self.gt_dir = os.path.join(root_dir, subset, "gt")
        self.img_names = os.listdir(self.img_dir)
        self.img_names.sort()
        if subset == 'train':
            assert len(self.img_names) == 861
        else:
            assert len(self.img_names) == 58

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        gt_name = img_name.split('_')[0] + '_clean.png'
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, gt_name)
        # 读取图像
        image = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        # Resizing images to multiples of 16 for whole-image restoration
        wd_new, ht_new = image.size
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        image = image.resize((wd_new, ht_new))
        gt = gt.resize((wd_new, ht_new))
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


class RaindropDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str = "../datasets/RainDrop", batch_size=1, pin_mem=True, num_workers=4):
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
            self.train_dataset = RaindropDataset(root_dir=self.root_dir, subset="train", transform=train_transform)
            self.val_dataset = RaindropDataset(root_dir=self.root_dir, subset="test_a", transform=test_transform)
        # Assign test dataset for use in dataloader(s)
        if stage == "validate":
            self.val_dataset = RaindropDataset(root_dir=self.root_dir, subset="test_a", transform=test_transform)
        if stage == "test":
            self.test_dataset = RaindropDataset(root_dir=self.root_dir, subset="test_a", transform=test_transform)

    # 以下方法创建不同阶段的数据加载器
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, drop_last=True,
                                           batch_size=self.batch_size, pin_memory=self.pin_mem,
                                           num_workers=self.num_workers, persistent_workers=True)

    # batch_size must be 1 because size is different
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, drop_last=False,
                                           batch_size=1, pin_memory=self.pin_mem,
                                           num_workers=self.num_workers, persistent_workers=True)

    # batch_size must be 1 because size is different
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, shuffle=False, drop_last=False,
                                           batch_size=1, pin_memory=self.pin_mem,
                                           num_workers=self.num_workers, persistent_workers=True)

# # test code
# if __name__ == '__main__':
#     my_dataset = RainDropDataset(root_dir='../../datasets/Raindrop')
#     input, target = my_dataset.__getitem__(0)
#
#     # 展示图像和掩码
#     visualize(input, target)
#     plt.show()
