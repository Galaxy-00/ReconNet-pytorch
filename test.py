# %%
import numpy
import cv2
import os
from utils import compute_ssim, psnr

# %%
img1 = cv2.imread('./test1.png', 0)
img2 = cv2.imread('./test1_reconstructed.png')
cv2.imwrite('test1.png', img1)

# %%
recon_md_image_path = "./results/reconNet_MD/"
recon_m_image_path = "./results/reconNet_M/"
recon_image_path = "./results/reconNet/"
org_image_path = "./dataset/set11_png/"

SSIM, PSNR = [], []
SSIM_M, PSNR_M = [], []
SSIM_MD, PSNR_MD = [], []
for _, _, files in os.walk(org_image_path):
    for file in files:
        org_iamge = cv2.imread(os.path.join(org_image_path, file), 0)
        recon_md_image = cv2.imread(
            os.path.join(recon_md_image_path,
                         file[:-4] + "_reconstructed" + file[-4:]), 0)
        recon_m_image = cv2.imread(
            os.path.join(recon_m_image_path,
                         file[:-4] + "_reconstructed" + file[-4:]), 0)
        recon_image = cv2.imread(
            os.path.join(recon_image_path,
                         file[:-4] + "_reconstructed" + file[-4:]), 0)

        ssim_o = compute_ssim(org_iamge, recon_image)
        ssim_m = compute_ssim(org_iamge, recon_m_image)
        ssim_md = compute_ssim(org_iamge, recon_md_image)
        psnr_o = psnr(org_iamge, recon_image)
        psnr_m = psnr(org_iamge, recon_m_image)
        psnr_md = psnr(org_iamge, recon_md_image)
        SSIM.append(ssim_o)
        SSIM_M.append(ssim_m)
        SSIM_MD.append(ssim_md)
        PSNR.append(psnr_o)
        PSNR_M.append(psnr_m)
        PSNR_MD.append(psnr_md)
        print(
            '{} \nReconNet: ssim:{:.5f} psnr:{:.5f}\nReconNet_M: ssim:{:.5f} psnr:{:.5f}\nReconNet_MD: ssim:{:.5f} psnr:{:.5f}'
            .format(file, ssim_o, psnr_o, ssim_m, psnr_m, ssim_md, psnr_md))

print(
    "Average\nReconNet:  ssim:{:.5f} psnr:{:.5f}\nReconNet_M:  ssim:{:.5f} psnr:{:.5f}\nReconNet_MD:  ssim:{:.5f} psnr:{:.5f}"
    .format(numpy.mean(SSIM), numpy.mean(PSNR), numpy.mean(SSIM_M),
            numpy.mean(PSNR_M), numpy.mean(SSIM_MD), numpy.mean(PSNR_MD)))

# %%
