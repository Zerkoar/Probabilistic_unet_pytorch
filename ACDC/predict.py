import pylab
from matplotlib import pyplot as plt

from probabilistic_unet import ProbabilisticUnet
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_ACDC import ACDC_Date

dataset = ACDC_Date()
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler, pin_memory=True)
test_loader = DataLoader(dataset, batch_size=15, sampler=test_sampler)
Iou = []

def Predict_result():
    global model, img, label, state_dict, elbo, intersection, union
    model = ProbabilisticUnet(input_channels=1, num_classes=4, num_filters=[32, 64, 128, 192], latent_dim=2,
                              no_convs_fcomb=4,
                              beta=10.0)
    model.to('cuda')
    img, label = next(iter(test_loader))
    img = img.to('cuda')
    label = label.to('cuda')
    label = torch.unsqueeze(label, 1)
    state_dict = torch.load('./weights/epoch_96,Iou_0.8961,test_Iou_0.8328.pth')
    model.load_state_dict(state_dict)
    model.eval()
    # torch.Size([batch_size, 2, 128, 128])
    model.forward(img, label)
    elbo, pred = model.elbo(label)
    pred = torch.argmax(pred, dim=1)
    pred = torch.unsqueeze(pred, dim=1)
    plt.figure(figsize=(25, 8))
    column = 3
    for i in range(pred.shape[0]):
    #     if np.max(label[i].cpu().numpy()) > 0 and np.max(pred[i].cpu().numpy()) > 0:
    #         # intersection = torch.logical_and(torch.squeeze(label), torch.argmax(pred, dim=1))
    #
    #         intersection = torch.logical_and(label[i], pred[i])
    #         union = torch.logical_or(label[i], pred[i])
    #         batch_iou = torch.sum(intersection) / torch.sum(union)
    #         Iou.append(batch_iou.item())

        plt.subplot(column, 15, i + 1)
        plt.imshow(img[i].permute(1, 2, 0).cpu().numpy())
        plt.subplot(column, 15, i + 16)
        plt.title("label")
        plt.imshow(label[i].permute(1, 2, 0).cpu().numpy())
        plt.subplot(column, 15, i + 31)
        plt.title("predict")
        plt.imshow(pred[i].permute(1, 2, 0).cpu().detach().numpy())
    pylab.show()

if __name__ == '__main__':
    # for i in range(4):
    Predict_result()
    # print(round(np.mean(Iou), 4))