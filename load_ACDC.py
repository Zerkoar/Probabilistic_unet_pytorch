import glob

import numpy as np
import pylab
import torch
from PIL import Image
from matplotlib import pyplot as plt
from pydicom import dataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms


class ACDC_Date(Dataset):
    def __init__(self):
        root = glob.glob("./ACDCtrainimg/*")
        imgs_path = []
        labels_path = []

        for i in root:
            if i.split("_")[-1] != "gt":
                imgs_path.append(i)
            else:
                labels_path.append(i)

        imgs = []
        labels = []

        for img, label in zip(imgs_path, labels_path):
            img_path = glob.glob(img + "/*")
            label_path = glob.glob(label + "/*")
            for i, l in zip(img_path, label_path):
                # 去除无病灶和分类数超过四的数据
                if np.max(np.array(Image.open(l))).all() > 0 and np.unique(np.array(Image.open(l))).all() < 5:
                    imgs.append(i)
                    labels.append(l)

        self.img = imgs
        self.mask = labels
        # self.transform = transform
        self.idx = {0: 0, 1: 85, 2: 170, 3: 255}

    def __getitem__(self, index):
        img = self.img[index]
        mask = self.mask[index]
        img_open = Image.open(img)
        # img_tensor = self.transform(img_open)
        img_tensor = transforms.ToTensor()(img_open)
        img_tensor = transforms.CenterCrop((128, 128))(img_tensor)

        mask_open = Image.open(mask)
        # mask_tensor = self.transform(mask_open)
        mask_np = np.array(mask_open)
        # 将像素值替换成标签
        mask = mask_np.copy()
        for k, v in self.idx.items():
            mask_np[mask == v] = k
        # 将字典中未包含的像素值转成 0
        mask_np = np.where(mask_np > 4, 0, mask_np)

        mask_tensor = torch.from_numpy(mask_np)
        # a = torch.unique(mask_tensor)
        mask_tensor = transforms.CenterCrop((128, 128))(mask_tensor)
        mask_tensor = torch.squeeze(mask_tensor).type(torch.float)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.img)


# if __name__ == '__main__':
#     dataset = ACDC_Date()
#     dataset_size = len(dataset)
#     indices = list(range(dataset_size))
#     split = int(np.floor(0.1 * dataset_size))
#     np.random.shuffle(indices)
#     train_indices, test_indices = indices[split:], indices[:split]
#     train_sampler = SubsetRandomSampler(train_indices)
#     test_sampler = SubsetRandomSampler(test_indices)
#     train_loader = DataLoader(dataset, batch_size=5, sampler=train_sampler)
#     test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
#
#     i, l = next(iter(train_loader))
#     plt.figure(figsize=(12, 12))
#     plt.subplot(1, 3, 1)
#     plt.imshow(i[0].cpu().numpy())
#     plt.subplot(1, 3, 2)
#     plt.imshow(l[0].cpu().numpy())
#     pylab.show()