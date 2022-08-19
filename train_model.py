import pylab
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
from tqdm import tqdm
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location='./Data/')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=5, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=2, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices), len(test_indices)))

net = ProbabilisticUnet(input_channels=1, num_classes=2, num_filters=[32, 64, 128, 192], latent_dim=2, no_convs_fcomb=4,
                        beta=10.0)
net.to(device)

# 加载权重训练
# state_dict = torch.load('./weights/epoch_49,loss_0.0014,Iou_0.6596,test_Iou_0.5996.pth')
# net.load_state_dict(state_dict)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 50
plt_iou = []
# loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    epoch_iou = []
    running_loss = 0
    elob_loss = 0
    rec_loss = 0
    for step, (patch, mask, _) in enumerate(tqdm(train_loader)):
        patch, mask = patch.to(device), mask.to(device)
        mask = torch.unsqueeze(mask, 1)
        net.forward(patch, mask, training=True)
        elbo, y_pred = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = -elbo + 1e-5 * reg_loss
        # loss = loss_fn(y_pred, torch.squeeze(mask).type(torch.long))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            running_loss += loss.item()
            elob_loss += elbo.item()
            rec_loss += reg_loss.item()

            intersection = torch.logical_and(torch.squeeze(mask), y_pred)
            union = torch.logical_or(torch.squeeze(mask), y_pred)
            batch_iou = torch.sum(intersection).type(torch.float) / torch.sum(union).type(torch.float)
            epoch_iou.append(batch_iou.item())

    epoch_loss = running_loss / len(train_indices)

    epoch_test_iou = []
    net.eval()
    with torch.no_grad():
        for step, (patch, mask, _) in enumerate(tqdm(test_loader)):
            patch, mask = patch.to(device), mask.to(device)
            mask = torch.unsqueeze(mask, 1)
            net.forward(patch, mask, training=True)
            elbo, y_pred = net.elbo(mask)

            y_pred = torch.argmax(y_pred, dim=1)
            intersection = torch.logical_and(torch.squeeze(mask), y_pred)
            union = torch.logical_or(torch.squeeze(mask), y_pred)
            batch_iou = torch.sum(intersection).type(torch.float) / torch.sum(union).type(torch.float)
            epoch_test_iou.append(batch_iou.item())

    #         for i in range(y_pred.shape[0]):
    #             if np.max(mask[i].cpu().numpy()) > 0 and np.max(y_pred[i].cpu().numpy()) > 0:
    #                 intersection = torch.logical_and(torch.squeeze(mask[i]), y_pred[i])
    #                 union = torch.logical_or(torch.squeeze(mask[i]), y_pred[i])
    #                 batch_iou = torch.sum(intersection).type(torch.float) / torch.sum(union).type(torch.float)
    #                 epoch_test_iou.append(batch_iou.item())

    print('epoch:', epoch, 'sum_loss:', round(epoch_loss, 4),
          'elbo:', round(-elob_loss / len(train_indices), 4),
          'running_loss:', round(running_loss, 4),
          'reg_loss:', round(rec_loss / len(train_indices), 4),
          'Iou:', round(np.mean(epoch_iou), 4),
          'test_Iou:', round(np.mean(epoch_test_iou), 4)
          )
    plt_iou.append(round(np.mean(epoch_test_iou), 4))
    static_dict = net.state_dict()
    torch.save(static_dict, './weights/epoch_{},loss_{},Iou_{},test_Iou_{}.pth'
               .format(epoch,
                       round(epoch_loss, 4),
                       round(np.mean(epoch_iou), 4),
                       round(np.mean(epoch_test_iou), 4)
                       ))

plt.plot(range(1, epochs + 1), plt_iou, label='Iou')
plt.legend()
plt.savefig('./my_figure.png')
pylab.show()
