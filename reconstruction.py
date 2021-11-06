# %%
import torch
import numpy as np
from torchvision import transforms
from torch import nn
from ReconNet import *
from ReconNet_M import ReconNet_M
from ReconNet_MD import ReconNet_MD
from dataset import TestDataset_M
import cv2
from matplotlib import pyplot as plt
from torch.utils import data


# %%
def test_reconstruction(model,
                        image_name,
                        CS_ratio,
                        device,
                        is_sampling=False):
    image_blocks = []
    test_img = plt.imread(image_name)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2YCR_CB)
    test_img = cv2.normalize(test_img,
                             None,
                             alpha=0,
                             beta=1,
                             norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_32F)
    test_img = test_img[:, :, 0]  # 提取亮度分量
    h, w = test_img.shape
    stride = 12
    filter_size = 33
    h_n = ((h - filter_size) // stride) + 2
    w_n = ((w - filter_size) // stride) + 2
    pad = np.zeros((h_n * stride + filter_size, w_n * stride + filter_size))
    pad[0:h, 0:w] = test_img[:][:]
    test_img = pad

    # 分块
    for i in range(h_n):
        for j in range(w_n):
            blocks = test_img[i * stride:(i * stride) + filter_size,
                              j * stride:(j * stride) + filter_size]
            image_blocks.append(blocks)

    image_blocks = np.array(image_blocks)
    transformations = transforms.Compose([transforms.ToTensor()])
    if (is_sampling):
        sampling_matrix = torch.from_numpy(
            load_sampling_matrix(CS_ratio)).float()
        test_data = TestDataset(image_blocks, sampling_matrix, transformations,
                                CS_ratio / 100)
    else:
        test_data = TestDataset_M(image_blocks, transformations)
    test_dl = data.DataLoader(test_data, h_n * w_n)

    # 模型输出
    feature, labels = next(iter(test_dl))
    feature, labels = feature.to(device), labels.to(device)
    feature, labels = feature.float(), labels.float()
    model = model.to(device)

    out = model(feature)
    out = out.view(labels.size())
    loss = nn.MSELoss()
    l = loss(out, labels)
    print('Test Loss', l.item())

    # 重建图片
    recon_img = torch.zeros(test_img.shape)
    k = 0
    # print('out', out.shape)
    # print('test_img', test_img.shape)
    for i in range(h_n):
        for j in range(w_n):
            recon_img[i * stride:(i * stride) + filter_size,
                      j * stride:(j * stride) +
                      filter_size] = out[k].cpu().reshape(33, 33)
            k += 1

    recon_img = recon_img.detach().numpy()
    recon_img[recon_img < 0] = 0
    recon_img[recon_img > 1] = 1
    recon_img = recon_img[:h, :w]

    recon_img = cv2.cvtColor(recon_img, cv2.COLOR_BGR2RGB)
    plt.imshow(recon_img)
    # 保存图片
    cv2.imwrite(image_name[:-4] + '_reconstructed' + image_name[-4:],
                recon_img * 255)
    # plt.savefig(image_name[:-4] + '_reconstructed' + image_name[-4],
    #             bbox_inches='tight')


# %%
if __name__ == '__main__':
    CS_ratio = 25
    device = torch.device('cuda')
    simpling_matrix = load_sampling_matrix(CS_ratio)
    reconNet = ReconNet_M(simpling_matrix, CS_ratio)
    reconNet.load_state_dict(torch.load('recon_m_{}.pkl'.format(CS_ratio)))
    # %%
    image_path = "./dataset/set11_png/"
    for _, _, files in os.walk(image_path):
        for file in files:
            print(file)
            image_name = os.path.join(image_path, file)
            test_reconstruction(reconNet, image_name, CS_ratio, device, is_sampling=False)
    # %%
    image_name = 'house.png'
    test_reconstruction(reconNet, image_name, CS_ratio, device, is_sampling=False)