import pylab
from matplotlib import pyplot as plt

from probabilistic_unet import ProbabilisticUnet
from load_LIDC_data import LIDC_IDRI
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


dataset = LIDC_IDRI(dataset_location='./Data/')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler, pin_memory=True)
test_loader = DataLoader(dataset, batch_size=4, sampler=test_sampler)
Iou = []

def Predict_result():
    global model, img, label, state_dict, elbo, intersection, union
    model = ProbabilisticUnet(input_channels=1, num_classes=2, num_filters=[32, 64, 128, 192], latent_dim=2,
                              no_convs_fcomb=4,
                              beta=10.0)
    model.to('cuda')
    img, label, _ = next(iter(test_loader))
    img = img.to('cuda')
    label = label.to('cuda')
    label = torch.unsqueeze(label, 1)
    state_dict = torch.load('./weights/epoch_49,loss_0.0018,Iou_0.7228,test_Iou_.pth')
    model.load_state_dict(state_dict)
    model.eval()
    # torch.Size([batch_size, 2, 128, 128])
    model.forward(img, label)
    elbo, pred = model.elbo(label)
    pred = torch.argmax(pred, dim=1)
    pred = torch.unsqueeze(pred, dim=1)
    for i in range(pred.shape[0]):
        if np.max(label[i].cpu().numpy()) > 0 and np.max(pred[i].cpu().numpy()) > 0:
            # intersection = torch.logical_and(torch.squeeze(label), torch.argmax(pred, dim=1))

            intersection = torch.logical_and(label[i], pred[i])
            union = torch.logical_or(label[i], pred[i])
            batch_iou = torch.sum(intersection) / torch.sum(union)
            Iou.append(batch_iou.item())

        plt.figure(figsize=(12, 12))
        plt.subplot(1, 3, 1)
        plt.imshow(img[i].permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 3, 2)
        plt.imshow(label[i].permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 3, 3)
        plt.imshow(pred[i].permute(1, 2, 0).cpu().detach().numpy())
        pylab.show()
    # print(pred.shape)
    # column = 5
    # plt.figure(figsize=(12, 12))
    # for i in range(5):
    #     # plt.title("真实图像")
    #     plt.subplot(5, column, i * column + 1)
    #     plt.imshow(img[i].permute(1, 2, 0).cpu().numpy())
    #
    #     # plt.title("真实标签")
    #     plt.subplot(5, column, i * column + 2)
    #     plt.imshow(label[i].permute(1, 2, 0).cpu().numpy())
    #
    #     # plt.title("分割结果")
    #     plt.subplot(5, column, i * column + 3)
    #     plt.imshow(torch.argmax(pred[i].permute(1, 2, 0), dim=-1).cpu().detach().numpy())
    #     a = torch.argmax(pred[i].permute(1, 2, 0), dim=-1).cpu().detach().numpy()
    #
    #     # plt.subplot(5, column, i * column + 4)
    #     # plt.imshow(pred[i][0].cpu().detach().numpy())
    #     # plt.subplot(5, column, i * column + 5)
    #     # plt.imshow(pred[i][1].cpu().detach().numpy())
    #
    # pylab.show()


if __name__ == '__main__':
    Predict_result()
    print(round(np.mean(Iou), 4))