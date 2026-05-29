import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ======================================================
# CONFIGURACIÓN
# ======================================================

INPUT_DIR = "data/processed/inputs"
TARGET_DIR = "data/processed/targets"

MODEL_DIR = "models"
RESULT_DIR = "training_results"

MODEL_PATH = os.path.join(MODEL_DIR, "unet_ct_residual_conservative1.pth")

IMG_SIZE = 256
BATCH_SIZE = 4

# Menos épocas para evitar que aprenda a suavizar demasiado
EPOCHS = 15

LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8
SEED = 42

# Pérdida de bordes: obliga a preservar contornos
GRADIENT_LOSS_WEIGHT = 1.0

# Pérdida de identidad: evita que la red se aleje demasiado de la FBP
IDENTITY_LOSS_WEIGHT = 0.6

# Escala residual baja: la red solo puede hacer correcciones pequeñas
RESIDUAL_SCALE = 0.03


# ======================================================
# REPRODUCIBILIDAD
# ======================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================
# DATASET
# ======================================================

class CTEnhancementDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".npy")
        ])

        self.target_files = sorted([
            os.path.join(target_dir, f)
            for f in os.listdir(target_dir)
            if f.endswith(".npy")
        ])

        if len(self.input_files) == 0:
            raise RuntimeError("No hay archivos en data/processed/inputs")

        if len(self.target_files) == 0:
            raise RuntimeError("No hay archivos en data/processed/targets")

        if len(self.input_files) != len(self.target_files):
            raise RuntimeError(
                f"No coincide número de inputs y targets: "
                f"{len(self.input_files)} vs {len(self.target_files)}"
            )

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        x = np.load(self.input_files[idx]).astype(np.float32)
        y = np.load(self.target_files[idx]).astype(np.float32)

        x = np.nan_to_num(x)
        y = np.nan_to_num(y)

        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)

        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)

        return x, y


# ======================================================
# MODELO U-NET RESIDUAL CONSERVADOR
# ======================================================

class DoubleConv(nn.Module):
    """
    Sin BatchNorm para evitar artefactos con datasets pequeños
    y batch sizes bajos.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualUNetConservative(nn.Module):
    def __init__(self, residual_scale=0.05):
        super().__init__()

        self.residual_scale = residual_scale

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

        # Corrección residual pequeña
        correction = torch.tanh(self.out(d1)) * self.residual_scale

        # Resultado final: FBP original + corrección limitada
        return torch.clamp(x + correction, 0, 1)


# ======================================================
# PÉRDIDAS Y MÉTRICAS
# ======================================================

def gradient_loss(pred, target):
    """
    Compara gradientes de predicción y target.
    Ayuda a conservar bordes y estructuras finas.
    """
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

    loss_x = torch.mean(torch.abs(pred_dx - target_dx))
    loss_y = torch.mean(torch.abs(pred_dy - target_dy))

    return loss_x + loss_y


def calcular_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)

    if mse.item() == 0:
        return 99.0

    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# ======================================================
# VISUALIZACIÓN
# ======================================================

def guardar_ejemplo(model, dataset, device, epoch):
    model.eval()

    os.makedirs(RESULT_DIR, exist_ok=True)

    idx = random.randint(0, len(dataset) - 1)

    x, y = dataset[idx]
    x_in = x.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x_in).cpu().squeeze().numpy()

    x_np = x.squeeze().numpy()
    y_np = y.squeeze().numpy()

    diff_fbp = np.abs(y_np - x_np)
    diff_ia = np.abs(y_np - pred)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 5, 1)
    plt.imshow(x_np, cmap="gray")
    plt.title("Input FBP")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(pred, cmap="gray")
    plt.title("IA conservadora")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(y_np, cmap="gray")
    plt.title("Target DICOM")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(diff_fbp, cmap="magma")
    plt.title("|Target - FBP|")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(diff_ia, cmap="magma")
    plt.title("|Target - IA|")
    plt.axis("off")

    plt.tight_layout()

    out_path = os.path.join(RESULT_DIR, f"epoch_{epoch:03d}_conservative.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


# ======================================================
# ENTRENAMIENTO
# ======================================================

def train():
    set_seed(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo usado: {device}")

    dataset = CTEnhancementDataset(INPUT_DIR, TARGET_DIR)

    total = len(dataset)
    train_size = int(TRAIN_SPLIT * total)
    val_size = total - train_size

    if val_size == 0:
        train_size = total - 1
        val_size = 1

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print(f"Total de pares: {total}")
    print(f"Entrenamiento: {train_size}")
    print(f"Validación: {val_size}")
    print(f"Residual scale: {RESIDUAL_SCALE}")
    print(f"Gradient weight: {GRADIENT_LOSS_WEIGHT}")
    print(f"Identity weight: {IDENTITY_LOSS_WEIGHT}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ResidualUNetConservative(
        residual_scale=RESIDUAL_SCALE
    ).to(device)

    l1_loss = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):

        # ------------------------------
        # ENTRENAMIENTO
        # ------------------------------
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_grad = 0.0
        train_identity = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss_l1 = l1_loss(pred, y)
            loss_grad = gradient_loss(pred, y)
            loss_identity = l1_loss(pred, x)

            loss = (
                loss_l1
                + GRADIENT_LOSS_WEIGHT * loss_grad
                + IDENTITY_LOSS_WEIGHT * loss_identity
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_l1 += loss_l1.item()
            train_grad += loss_grad.item()
            train_identity += loss_identity.item()

        train_loss /= len(train_loader)
        train_l1 /= len(train_loader)
        train_grad /= len(train_loader)
        train_identity /= len(train_loader)

        # ------------------------------
        # VALIDACIÓN
        # ------------------------------
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        val_grad = 0.0
        val_identity = 0.0
        val_psnr = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)

                loss_l1 = l1_loss(pred, y)
                loss_grad = gradient_loss(pred, y)
                loss_identity = l1_loss(pred, x)

                loss = (
                    loss_l1
                    + GRADIENT_LOSS_WEIGHT * loss_grad
                    + IDENTITY_LOSS_WEIGHT * loss_identity
                )

                val_loss += loss.item()
                val_l1 += loss_l1.item()
                val_grad += loss_grad.item()
                val_identity += loss_identity.item()
                val_psnr += calcular_psnr(pred, y).item()

        val_loss /= len(val_loader)
        val_l1 /= len(val_loader)
        val_grad /= len(val_loader)
        val_identity /= len(val_loader)
        val_psnr /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.5f} "
            f"(L1={train_l1:.5f}, Grad={train_grad:.5f}, Id={train_identity:.5f}) | "
            f"Val Loss: {val_loss:.5f} "
            f"(L1={val_l1:.5f}, Grad={val_grad:.5f}, Id={val_identity:.5f}) | "
            f"Val PSNR: {val_psnr:.2f} dB"
        )

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Mejor modelo guardado: {MODEL_PATH}")

        # Guardar ejemplo visual
        if epoch == 1 or epoch % 5 == 0:
            guardar_ejemplo(model, dataset, device, epoch)

    # ------------------------------
    # Curva de pérdida
    # ------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Curva de entrenamiento U-Net residual conservadora")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(RESULT_DIR, "loss_curve_conservative.png"), dpi=150)
    plt.close()

    print("\nEntrenamiento terminado.")
    print(f"Mejor Val Loss: {best_val_loss:.5f}")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Resultados guardados en: {RESULT_DIR}")


if __name__ == "__main__":
    train()