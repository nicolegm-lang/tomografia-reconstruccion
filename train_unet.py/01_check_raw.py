import os
import numpy as np
import pydicom


RAW_DIR = "data/raw"


def get_instance_number(dcm_path):
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return int(getattr(ds, "InstanceNumber", 0))
    except Exception:
        return 0


def main():
    print("Revisando carpeta raw...\n")

    pacientes = [
        p for p in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, p))
    ]

    if not pacientes:
        print("No encontré carpetas de pacientes en data/raw")
        return

    for paciente in pacientes:
        carpeta = os.path.join(RAW_DIR, paciente)

        npy_files = [
            f for f in os.listdir(carpeta)
            if f.lower().endswith(".npy")
        ]

        dicom_dir = os.path.join(carpeta, "dicom")

        print("=" * 60)
        print(f"Paciente: {paciente}")

        if not npy_files:
            print("No encontré archivo .npy de sinograma.")
            continue

        sino_path = os.path.join(carpeta, npy_files[0])
        sino = np.load(sino_path)

        print(f"Sinograma: {npy_files[0]}")
        print(f"Forma del sinograma: {sino.shape}")

        if not os.path.isdir(dicom_dir):
            print("No encontré carpeta dicom/")
            continue

        dicoms = [
            os.path.join(dicom_dir, f)
            for f in os.listdir(dicom_dir)
            if f.lower().endswith(".dcm")
        ]

        dicoms = sorted(dicoms, key=get_instance_number)

        print(f"Número de DICOM: {len(dicoms)}")

        if len(dicoms) > 0:
            ds = pydicom.dcmread(dicoms[0], stop_before_pixels=True)
            print(f"Primer InstanceNumber: {getattr(ds, 'InstanceNumber', 'N/A')}")
            print(f"Rows x Columns: {getattr(ds, 'Rows', 'N/A')} x {getattr(ds, 'Columns', 'N/A')}")
            print(f"StudyDescription: {getattr(ds, 'StudyDescription', 'N/A')}")
            print(f"SeriesDescription: {getattr(ds, 'SeriesDescription', 'N/A')}")

        if len(sino.shape) == 3:
            n_cortes = min(sino.shape)
            print(f"Posibles cortes en sinograma: {n_cortes}")

            if n_cortes == len(dicoms):
                print("Coinciden cortes de sinograma y DICOM.")
            else:
                print("No coinciden exactamente. Se usará el mínimo disponible.")

    print("\nRevisión terminada.")


if __name__ == "__main__":
    main()