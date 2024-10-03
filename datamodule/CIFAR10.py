# https://github.com/ermongroup/ddim/blob/main/datasets/__init__.py
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision.transforms as transforms
from pytorch_lightning.utilities import CombinedLoader

from .ImgGenerator import ImgGenerator

train_transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [transforms.Resize(32), transforms.ToTensor()]
)


class CIFAR10_Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.dataset = CIFAR10(root_dir, train=train, download=True, transform=transform)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        image, label = self.dataset.__getitem__(idx)
        return image


class CIFAR10_DataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str = "../datasets/cifar10", batch_size=1, pin_mem=True, num_workers=4):
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
            self.train_dataset = CIFAR10_Dataset(self.root_dir, train=True, transform=train_transform)
            # following DDPM
            self.test_dataset = CIFAR10_Dataset(self.root_dir, train=True, transform=test_transform)
        # Assign test dataset for use in dataloader(s)
        if stage == "validate":
            self.test_dataset = CIFAR10_Dataset(self.root_dir, train=True, transform=test_transform)
        if stage == "test":
            self.test_dataset = CIFAR10_Dataset(self.root_dir, train=True, transform=test_transform)

    # 以下方法创建不同阶段的数据加载器
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, drop_last=True,
                          batch_size=self.batch_size, pin_memory=self.pin_mem,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        # 50K validation
        val_iterables = {
            'pred_images': DataLoader(ImgGenerator(generate_num=50000, generate_size=(3, 32, 32)),
                                      shuffle=False, drop_last=False,
                                      batch_size=self.batch_size, pin_memory=self.pin_mem,
                                      num_workers=self.num_workers,
                                      persistent_workers=True),
            'true_images': DataLoader(self.test_dataset, shuffle=False, drop_last=False,
                                      batch_size=self.batch_size, pin_memory=self.pin_mem,
                                      num_workers=self.num_workers,
                                      persistent_workers=True),
        }
        val_combined_loader = CombinedLoader(val_iterables, 'max_size')
        return val_combined_loader

    def test_dataloader(self):
        # 50K test
        test_iterables = {
            'pred_images': DataLoader(ImgGenerator(generate_num=50000, generate_size=(3, 32, 32)),
                                      shuffle=False, drop_last=False,
                                      batch_size=self.batch_size,
                                      pin_memory=self.pin_mem,
                                      num_workers=self.num_workers,
                                      persistent_workers=True),
            'true_images': DataLoader(self.test_dataset, shuffle=False, drop_last=False,
                                      batch_size=self.batch_size,
                                      pin_memory=self.pin_mem,
                                      num_workers=self.num_workers,
                                      persistent_workers=True),
        }
        test_combined_loader = CombinedLoader(test_iterables, 'max_size')
        return test_combined_loader


# # test code
# if __name__ == '__main__':
#     my_dataset = CIFAR10_Dataset(root_dir='../../datasets/cifar10', transform=train_transform)
#     print(my_dataset.__getitem__(0))
