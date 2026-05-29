import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# ======================================================
# CONFIGURACIÓN
# ======================================================

INPUT_DIR = "data/processed/inputs"
TARGET_DIR = "data/processed/targets"
MODEL_PATH = "models/unet_ct_residual_conservative1.pth"
OUT_DIR = "test_results"

NUM_EXAMPLES = 6


# ======================================================
# MODELO
# ======================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return torch.sigmoid(self.out(d1))


# ======================================================
# UTILIDADES
# ======================================================

def normalizar(img):
    img = img.astype(np.float32)
    img = np.nan_to_num(img)

    img = img - img.min()
    img = img / (img.max() + 1e-8)

    return img


def calcular_metricas(img, target):
    img = normalizar(img)
    target = normalizar(target)

    mse = np.mean((img - target) ** 2)
    rmse = np.sqrt(mse)
    ssim_val = ssim(target, img, data_range=1.0)
    psnr_val = psnr(target, img, data_range=1.0)

    return rmse, ssim_val, psnr_val


def cargar_archivos():
    inputs = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".npy")
    ])

    targets = sorted([
        os.path.join(TARGET_DIR, f)
        for f in os.listdir(TARGET_DIR)
        if f.endswith(".npy")
    ])

    if len(inputs) != len(targets):
        raise RuntimeError("No coincide el número de inputs y targets.")

    return inputs, targets


# ======================================================
# TEST
# ======================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device)

    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    input_files, target_files = cargar_archivos()

    indices = list(range(len(input_files)))
    random.shuffle(indices)
    indices = indices[:min(NUM_EXAMPLES, len(indices))]

    metricas_fbp = []
    metricas_ia = []

    for k, idx in enumerate(indices):
        x = np.load(input_files[idx]).astype(np.float32)
        y = np.load(target_files[idx]).astype(np.float32)

        x = normalizar(x)
        y = normalizar(y)

        x_tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_tensor).cpu().squeeze().numpy()

        pred = normalizar(pred)

        rmse_fbp, ssim_fbp, psnr_fbp = calcular_metricas(x, y)
        rmse_ia, ssim_ia, psnr_ia = calcular_metricas(pred, y)

        metricas_fbp.append([rmse_fbp, ssim_fbp, psnr_fbp])
        metricas_ia.append([rmse_ia, ssim_ia, psnr_ia])

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(x, cmap="gray")
        plt.title(f"FBP\nSSIM={ssim_fbp:.3f}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(pred, cmap="gray")
        plt.title(f"IA mejorada\nSSIM={ssim_ia:.3f}")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(y, cmap="gray")
        plt.title("DICOM referencia")
        plt.axis("off")

        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"test_{k:03d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

        print("=" * 60)
        print(f"Ejemplo {k} | archivo index: {idx}")
        print(f"FBP -> RMSE={rmse_fbp:.4f}, SSIM={ssim_fbp:.4f}, PSNR={psnr_fbp:.2f}")
        print(f"IA  -> RMSE={rmse_ia:.4f}, SSIM={ssim_ia:.4f}, PSNR={psnr_ia:.2f}")

    metricas_fbp = np.array(metricas_fbp)
    metricas_ia = np.array(metricas_ia)

    print("\n" + "=" * 60)
    print("PROMEDIOS")
    print(f"FBP -> RMSE={metricas_fbp[:,0].mean():.4f}, SSIM={metricas_fbp[:,1].mean():.4f}, PSNR={metricas_fbp[:,2].mean():.2f}")
    print(f"IA  -> RMSE={metricas_ia[:,0].mean():.4f}, SSIM={metricas_ia[:,1].mean():.4f}, PSNR={metricas_ia[:,2].mean():.2f}")

    print(f"\nImágenes guardadas en: {OUT_DIR}")


if __name__ == "__main__":
    main()