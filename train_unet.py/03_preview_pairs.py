import os
import numpy as np
import matplotlib.pyplot as plt


INPUT_DIR = "data/processed/inputs"
TARGET_DIR = "data/processed/targets"


def main():
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

    n = min(len(inputs), len(targets), 6)

    if n == 0:
        print("No hay pares para visualizar.")
        return

    plt.figure(figsize=(10, 3 * n))

    for i in range(n):
        x = np.load(inputs[i])
        y = np.load(targets[i])

        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(x, cmap="gray")
        plt.title(f"Input FBP {i}")
        plt.axis("off")

        plt.subplot(n, 2, 2*i + 2)
        plt.imshow(y, cmap="gray")
        plt.title(f"Target DICOM {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()