# https://github.com/GuoLanqing/ShadowFormer
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


# def visualize(image, mask, gt, original_image=None, original_mask=None, original_gt=None):
#     fontsize = 18
#
#     if original_image is None and original_gt is None:
#         f, ax = plt.subplots(3, 1, figsize=(8, 8))
#         ax[0].imshow(image)
#         ax[1].imshow(mask)
#         ax[2].imshow(gt)
#     else:
#         f, ax = plt.subplots(3, 2, figsize=(8, 8))
#
#         ax[0, 0].imshow(original_image)
#         ax[0, 0].set_title('Original image', fontsize=fontsize)
#
#         ax[1, 0].imshow(original_mask)
#         ax[1, 0].set_title('Original mask', fontsize=fontsize)
#
#         ax[2, 0].imshow(original_gt)
#         ax[2, 0].set_title('Original gt', fontsize=fontsize)
#
#         ax[0, 1].imshow(image)
#         ax[0, 1].set_title('Transformed image', fontsize=fontsize)
#
#         ax[1, 1].imshow(mask)
#         ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
#
#         ax[2, 1].imshow(gt)
#         ax[2, 1].set_title('Transformed gt', fontsize=fontsize)
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


class ISTD_Dataset(Dataset):
    def __init__(self, root_dir, subset="train", transform=None):
        """
        初始化函数
        :param root_dir: 数据集的根目录
        :param subset: 使用的数据子集 ('train', 'test')
        :param transform: 应用于数据的转换
        """
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.img_dir = os.path.join(root_dir, subset, subset+"_A")
        self.mask_dir = os.path.join(root_dir, subset, subset+"_B")
        self.gt_dir = os.path.join(root_dir, subset, subset+"_C")
        self.img_names = os.listdir(self.img_dir)
        self.img_names.sort()
        if subset == 'train':
            assert len(self.img_names) == 1330
        else:
            assert len(self.img_names) == 540

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)
        # 读取图像
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        gt = Image.open(gt_path).convert("RGB")
        # 将PIL图像转换为NumPy数组
        image = np.array(image)
        mask = np.array(mask)
        gt = np.array(gt)
        # 将所有255的值转换为1
        mask = np.where(mask == 255, 1, 0)
        if self.transform:
            augmented = self.transform(image=image, gt_image=gt, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            gt = augmented['gt_image']
        img_input = image
        mask = mask
        target = gt
        return img_input, mask.long(), target


class ISTD_DataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str = "../datasets/ISTD", batch_size=1, pin_mem=True, num_workers=4):
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
            self.train_dataset = ISTD_Dataset(root_dir=self.root_dir, subset="train", transform=train_transform)
            self.val_dataset = ISTD_Dataset(root_dir=self.root_dir, subset="test", transform=test_transform)
        # Assign test dataset for use in dataloader(s)
        if stage == "validate":
            self.val_dataset = ISTD_Dataset(root_dir=self.root_dir, subset="test", transform=test_transform)
        if stage == "test":
            self.test_dataset = ISTD_Dataset(root_dir=self.root_dir, subset="test", transform=test_transform)

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
#     my_dataset = ISTD_Dataset(root_dir='../../datasets/ISTD')
#     input, mask, target = my_dataset.__getitem__(100)
#
#     augmented = my_transform_train(image=input, gt_image=target, mask=mask)
#     input_medium = augmented['image']
#     mask_medium = augmented['mask']
#     target_medium = augmented['gt_image']
#
#     # 展示图像和掩码
#     visualize(input_medium, mask_medium, target_medium, original_image=input, original_mask=mask, original_gt=target)
#     plt.show()
