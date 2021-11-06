# %%
import torch
import os
from torch import nn
from scipy import io
from dataset import *
import torch.nn.functional as F


# %%
class ReconNet(nn.Module):
    def __init__(self, measure_rates=0.25):
        super(ReconNet, self).__init__()
        self.fc1 = nn.Linear(int(measure_rates * 33 * 33),
                             33 * 33)  # (272, 1089)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, padding=5)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(1, 64, kernel_size=11, padding=5)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv6 = nn.Conv2d(32, 1, kernel_size=7, padding=3)

    def forward(self, X):
        """ X的形状(batch_size, 1, 1, measure_rates*33*33), 输出形状(batch_size, 33, 33)"""
        X = self.fc1(X)
        X = X.view(-1, 33, 33)
        X = torch.unsqueeze(X, dim=1)
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))
        X = F.relu(self.conv5(X))
        X = F.relu(self.conv6(X))
        return torch.squeeze(X, dim=1)


def load_sampling_matrix(CS_ratio):
    path = "dataset/sampling_matrix"
    data = io.loadmat(os.path.join(path,
                                   str(CS_ratio) + '.mat'))['sampling_matrix']
    return data


# # %%
# data_iter = dataset()
# n = 0
# for feature, label in data_iter:
#     # print(n)
#     n += 1
#     print(feature.shape, label.shape)
#     break

# # %%
# measure_rates = 0.25
# inputs = torch.rand((1, 33, 33))
# A = load_sampling_matrix(int(measure_rates * 100))
# A = torch.from_numpy(A).float()
# print(A.shape)
# print(inputs.reshape(-1, 1).shape)
# y = torch.transpose(torch.matmul(A, inputs.reshape(-1, 1)), 0, 1)
# print(y.shape)
# recon0 = ReconNet(measure_rates)
# outputs = recon0(y)
# print(outputs.shape)


# %%
# sampling_matrix(272, 1089), X(batch_size, 33, 33)
def sampling(CS_ratio, X, device):
    sampling_matrix = torch.from_numpy(load_sampling_matrix(CS_ratio)).float()
    sampling_matrix = torch.unsqueeze(sampling_matrix,
                                      dim=0).to(device)  # (1, 272, 1089)
    X = X.to(device)
    X = X.reshape(X.shape[0], -1, 1)  # (batch_size, 1089, 1)
    res = torch.matmul(sampling_matrix, X)  # (batch_size, 272, 1)
    res = torch.transpose(res, 1, 2)  # (batch_size, 1, 272)
    return res


def train(net, CS_ratio, train_iter, lr, num_epochs, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on:', device)
    net.to(device)
    loss = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        net.train()
        for i, (feature,
                label) in enumerate(train_iter):  # (batch_size, 33, 33)
            feature, label = feature.to(device), label.to(device)
            feature = sampling(CS_ratio, feature, device)
            feature = torch.unsqueeze(feature,
                                      dim=1)  # (batch_size, 1, 1, 272)
            opt.zero_grad()
            y_hat = net(feature)
            l = loss(y_hat, label)
            l.backward()
            opt.step()
            if (i % 25 == 0):
                output = "CS_ratio: %d [%02d/%02d] loss: %.4f " % (
                    CS_ratio, epoch + 1, batch_size * i, l)
                print(output)


# %%
if __name__ == '__main__':
    CS_ratio = 25
    lr = 0.001
    num_epochs = 10
    batch_size = 25
    device = torch.device('cuda')
    recon = ReconNet(CS_ratio / 100)
    train_dataset = dataset()
    train_iter = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    train(recon, CS_ratio, train_iter, lr, num_epochs, device)

    # %%
    torch.save(recon.state_dict(), 'recon_{}.pkl'.format(CS_ratio))

# # %%
# # ReconNet输入取样后的
# for features, labels in train_dataset:
#     test = torch.unsqueeze(feature, dim=0)
#     test = sampling(CS_ratio, test, device)
#     test = torch.unsqueeze(test, dim=1)
#     res = recon(test)
#     res = torch.squeeze(res, 0)
#     imsave('res.jpg', res.cpu().detach().numpy())
#     imsave('label.jpg', labels.cpu().detach().numpy())
#     break