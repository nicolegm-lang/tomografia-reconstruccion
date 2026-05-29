import os
import numpy as np
import pydicom

from skimage.transform import iradon, resize
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage.filters import unsharp_mask

from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift


# ======================================================
# CONFIGURACIÓN
# ======================================================

RAW_DIR = "data/raw"
OUT_INPUTS = "data/processed/inputs"
OUT_TARGETS = "data/processed/targets"

IMG_SIZE = 256

# Para prueba rápida usa 10.
# Cuando ya se vea bien, cambia a None.
MAX_SLICES_PER_PATIENT = None

# Si los cortes se ven desfasados, cambia esto a True.
INVERTIR_ORDEN_DICOM = False

# Parámetros de reconstrucción inicial
ORIENTACION_SINOGRAMA = "Ángulos × Detector"
RANGO_ANGULAR = "360°"
GEOMETRIA = "Haz de abanico"

# Ajusta si la reconstrucción sigue desplazada/deformada.
D_FUENTE = 6.0
SHIFT_CENTRO = 80

RECORTAR_DETECTOR = True
REDUCIR_360_A_180 = False
APLICAR_MASCARA = True
FORZAR_ORIENTACION_FINAL = "original"

# ======================================================
# FUNCIONES BÁSICAS
# ======================================================

def normalizar_percentil(img, p1=1, p99=99):
    img = img.astype(np.float32)
    img = np.nan_to_num(img)

    a, b = np.percentile(img, (p1, p99))

    if b - a < 1e-8:
        return np.zeros_like(img, dtype=np.float32)

    img = np.clip(img, a, b)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img.astype(np.float32)

def alinear_por_traslacion(input_img, target_img):
    """
    Alinea el input con el target usando registro por traslación.
    No rota ni deforma; solo desplaza la imagen para centrarla mejor.
    """
    input_n = normalizar_percentil(input_img, 1, 99)
    target_n = normalizar_percentil(target_img, 1, 99)

    try:
        shift_estimado, error, _ = phase_cross_correlation(
            target_n,
            input_n,
            upsample_factor=10
        )

        input_alineado = ndi_shift(
            input_n,
            shift=shift_estimado,
            mode="constant",
            cval=0
        )

        return normalizar_percentil(input_alineado, 1, 99), shift_estimado, error

    except Exception:
        return input_n, (0, 0), None

def get_instance_number(dcm_path):
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return int(getattr(ds, "InstanceNumber", 0))
    except Exception:
        return 0


def leer_dicom_hu(dcm_path):
    ds = pydicom.dcmread(dcm_path)

    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))

    img = img * slope + intercept

    return img


# ======================================================
# SINOGRAMA
# ======================================================

def extraer_corte_sinograma(stack, idx):
    """
    Extrae un corte 2D de un sinograma 3D.

    Ejemplo:
    stack = (984, 888, 39)
    El eje más pequeño suele ser el eje de cortes.
    """
    if stack.ndim == 2:
        return stack

    if stack.ndim != 3:
        raise ValueError(f"Sinograma con dimensión no soportada: {stack.shape}")

    eje_cortes = int(np.argmin(stack.shape))

    sino_2d = np.take(stack, idx, axis=eje_cortes)
    sino_2d = np.squeeze(sino_2d)

    if sino_2d.ndim != 2:
        raise ValueError(f"No se pudo extraer sinograma 2D: {sino_2d.shape}")

    return sino_2d


def preparar_orientacion_sinograma(sino, orientacion):
    """
    iradon espera:
    filas = detector
    columnas = ángulos
    """
    sino = np.asarray(sino, dtype=np.float32)

    if orientacion == "Detector × Ángulos":
        return sino

    if orientacion == "Ángulos × Detector":
        return sino.T

    # Automático
    if sino.shape[0] > sino.shape[1]:
        return sino.T

    return sino


def recortar_sinograma_util(sino, umbral_relativo=0.06, margen=12):
    perfil = np.mean(normalizar_percentil(sino, 1, 99), axis=1)
    indices = np.where(perfil > umbral_relativo)[0]

    if len(indices) == 0:
        return sino

    i0 = max(0, indices[0] - margen)
    i1 = min(sino.shape[0], indices[-1] + margen + 1)

    return sino[i0:i1, :]


def reducir_360_a_180_promediado(sino):
    n_ang = sino.shape[1]
    mitad = n_ang // 2

    if mitad < 4:
        return sino

    s1 = sino[:, :mitad]
    s2 = sino[:, mitad:mitad * 2]

    if s1.shape != s2.shape:
        return s1

    s2 = np.flipud(s2)

    return 0.5 * (s1 + s2)


def aplicar_mascara_circular(img):
    n, m = img.shape

    y, x = np.ogrid[:n, :m]
    cy, cx = n // 2, m // 2
    r = min(cy, cx)

    mascara = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

    out = np.zeros_like(img)
    out[mascara] = img[mascara]

    return out


def rebinning_fanbeam_a_paralelo(sino_fan, d_fuente=6.0, rango="360°"):
    """
    Rebinning aproximado fan-beam -> parallel-beam.
    """
    sino_fan = np.asarray(sino_fan, dtype=np.float32)

    n_det, n_ang = sino_fan.shape

    rango_grados = 360.0 if rango == "360°" else 180.0
    rango_rad = np.deg2rad(rango_grados)

    gamma_max = np.arctan(1.0 / d_fuente)
    t_max = d_fuente * np.sin(gamma_max)

    t_vals = np.linspace(-t_max, t_max, n_det)
    theta_vals = np.linspace(0.0, rango_rad, n_ang, endpoint=False)

    T, THETA = np.meshgrid(t_vals, theta_vals, indexing="ij")

    gamma = np.arcsin(np.clip(T / d_fuente, -0.999, 0.999))
    beta = (THETA - gamma) % rango_rad

    u = d_fuente * np.tan(gamma)

    fila = ((u + 1.0) / 2.0) * (n_det - 1)
    columna = (beta / rango_rad) * n_ang

    sino_paralelo = map_coordinates(
        sino_fan,
        [fila, columna],
        order=1,
        mode="wrap"
    )

    return normalizar_percentil(sino_paralelo, 0.5, 99.7)


def reconstruir_fbp_avanzado(sino_2d):
    """
    Genera el input del modelo.
    Debe verse como una CT degradada, no como una imagen totalmente deformada.
    """
    sino = normalizar_percentil(sino_2d, 0.5, 99.7)

    sino = preparar_orientacion_sinograma(
        sino,
        ORIENTACION_SINOGRAMA
    )

    sino = np.roll(sino, SHIFT_CENTRO, axis=0)

    rango_usado = RANGO_ANGULAR

    if GEOMETRIA == "Haz de abanico":
        sino = rebinning_fanbeam_a_paralelo(
            sino,
            d_fuente=D_FUENTE,
            rango=rango_usado
        )

    if RECORTAR_DETECTOR:
        sino = recortar_sinograma_util(sino)

    if REDUCIR_360_A_180 and rango_usado == "360°":
        sino = reducir_360_a_180_promediado(sino)
        rango_usado = "180°"

    sino = normalizar_percentil(sino, 0.5, 99.7)

    grados = 360.0 if rango_usado == "360°" else 180.0

    theta = np.linspace(
        0,
        grados,
        sino.shape[1],
        endpoint=False
    )

    rec = iradon(
        sino,
        theta=theta,
        filter_name="ramp",
        output_size=IMG_SIZE,
        circle=False
    )

    rec = normalizar_percentil(rec, 0.5, 99.5)

    # Realce suave para que el input tenga estructuras visibles
    rec = gaussian_filter(rec, sigma=0.2)
    rec = unsharp_mask(
        rec,
        radius=1.0,
        amount=0.4,
        preserve_range=True
    )

    rec = normalizar_percentil(rec, 0.5, 99.5)

    if APLICAR_MASCARA:
        rec = aplicar_mascara_circular(rec)

    return rec.astype(np.float32)


# ======================================================
# ORIENTACIÓN GLOBAL POR PACIENTE
# ======================================================

def aplicar_transformacion(img, nombre):
    if nombre == "original":
        return img
    elif nombre == "rot90":
        return np.rot90(img, 1)
    elif nombre == "rot180":
        return np.rot90(img, 2)
    elif nombre == "rot270":
        return np.rot90(img, 3)
    elif nombre == "flip_h":
        return np.fliplr(img)
    elif nombre == "flip_v":
        return np.flipud(img)
    elif nombre == "rot90_flip_h":
        return np.fliplr(np.rot90(img, 1))
    elif nombre == "rot90_flip_v":
        return np.flipud(np.rot90(img, 1))
    elif nombre == "rot180_flip_h":
        return np.fliplr(np.rot90(img, 2))
    elif nombre == "rot180_flip_v":
        return np.flipud(np.rot90(img, 2))
    elif nombre == "rot270_flip_h":
        return np.fliplr(np.rot90(img, 3))
    elif nombre == "rot270_flip_v":
        return np.flipud(np.rot90(img, 3))
    else:
        return img


def obtener_lista_transformaciones():
    return [
        "original",
        "rot90",
        "rot180",
        "rot270",
        "flip_h",
        "flip_v",
        "rot90_flip_h",
        "rot90_flip_v",
        "rot180_flip_h",
        "rot180_flip_v",
        "rot270_flip_h",
        "rot270_flip_v",
    ]


def encontrar_mejor_orientacion_global(stack, dicoms, n_prueba=5):
    """
    Busca una sola orientación para todo el paciente.
    Usa los primeros n_prueba cortes y elige la transformación
    con mejor SSIM promedio.
    """
    transformaciones = obtener_lista_transformaciones()
    scores = {t: [] for t in transformaciones}

    n = min(n_prueba, len(dicoms))

    for i in range(n):
        sino_2d = extraer_corte_sinograma(stack, i)
        fbp = reconstruir_fbp_avanzado(sino_2d)

        target = leer_dicom_hu(dicoms[i])
        target = resize(
            target,
            (IMG_SIZE, IMG_SIZE),
            anti_aliasing=True
        )
        target = normalizar_percentil(target, 1, 99)

        for t in transformaciones:
            var = aplicar_transformacion(fbp, t)

            if var.shape != target.shape:
                var = resize(
                    var,
                    target.shape,
                    anti_aliasing=True
                )

            var = normalizar_percentil(var, 1, 99)

            try:
                score = ssim(target, var, data_range=1.0)
            except Exception:
                score = -1.0

            scores[t].append(score)

    promedios = {
        t: np.mean(v) if len(v) > 0 else -1.0
        for t, v in scores.items()
    }

    mejor = max(promedios, key=promedios.get)

    return mejor, promedios[mejor], promedios


# ======================================================
# PROCESAMIENTO POR PACIENTE
# ======================================================

def procesar_paciente(paciente, contador_global):
    carpeta = os.path.join(RAW_DIR, paciente)

    npy_files = [
        f for f in os.listdir(carpeta)
        if f.lower().endswith(".npy")
    ]

    if not npy_files:
        print(f"{paciente}: no tiene archivo .npy")
        return contador_global

    sino_path = os.path.join(carpeta, npy_files[0])
    dicom_dir = os.path.join(carpeta, "dicom")

    if not os.path.isdir(dicom_dir):
        print(f"{paciente}: no tiene carpeta dicom/")
        return contador_global

    dicoms = [
        os.path.join(dicom_dir, f)
        for f in os.listdir(dicom_dir)
        if f.lower().endswith(".dcm")
    ]

    dicoms = sorted(dicoms, key=get_instance_number, reverse=INVERTIR_ORDEN_DICOM)

    if len(dicoms) == 0:
        print(f"{paciente}: no tiene archivos DICOM")
        return contador_global

    stack = np.load(sino_path)

    if stack.ndim == 3:
        n_cortes_sino = min(stack.shape)
    else:
        n_cortes_sino = 1

    n = min(n_cortes_sino, len(dicoms))

    if MAX_SLICES_PER_PATIENT is not None:
        n = min(n, MAX_SLICES_PER_PATIENT)

    print("=" * 70)
    print(f"Procesando paciente: {paciente}")
    print(f"Archivo sinograma: {npy_files[0]}")
    print(f"Forma sinograma: {stack.shape}")
    print(f"Número DICOM: {len(dicoms)}")
    print(f"Cortes a procesar: {n}")
    print(f"Geometría: {GEOMETRIA}")
    print(f"Orientación sinograma: {ORIENTACION_SINOGRAMA}")
    print(f"Rango: {RANGO_ANGULAR}")
    print(f"D: {D_FUENTE}")
    print(f"Shift: {SHIFT_CENTRO}")
    print(f"Invertir DICOM: {INVERTIR_ORDEN_DICOM}")

    if FORZAR_ORIENTACION_FINAL is not None:
        mejor_orientacion_global = FORZAR_ORIENTACION_FINAL
        score_global = 0
        print(f"Orientación final forzada: {mejor_orientacion_global}")
    else:
        mejor_orientacion_global, score_global, resumen_scores = encontrar_mejor_orientacion_global(
            stack,
            dicoms,
            n_prueba=min(5, n)
        )

        print(f"Mejor orientación global: {mejor_orientacion_global}")
        print(f"SSIM promedio global: {score_global:.4f}")

    for i in range(n):
        try:
            sino_2d = extraer_corte_sinograma(stack, i)

            input_fbp = reconstruir_fbp_avanzado(sino_2d)

            target = leer_dicom_hu(dicoms[i])
            target = resize(
                target,
                (IMG_SIZE, IMG_SIZE),
                anti_aliasing=True
            )
            target = normalizar_percentil(target, 1, 99)

            input_fbp_alineado = aplicar_transformacion(
                input_fbp,
                mejor_orientacion_global
            )

            input_fbp_alineado = normalizar_percentil(
                input_fbp_alineado,
                1,
                99
            )

            # Alineación fina por desplazamiento
            input_fbp_alineado, shift_xy, error_registro = alinear_por_traslacion(
                input_fbp_alineado,
                target
            )

            try:
                score_ssim = ssim(
                    target,
                    input_fbp_alineado,
                    data_range=1.0
                )
            except Exception:
                score_ssim = -1.0

            input_name = f"input_{contador_global:06d}.npy"
            target_name = f"target_{contador_global:06d}.npy"

            np.save(
                os.path.join(OUT_INPUTS, input_name),
                input_fbp_alineado.astype(np.float32)
            )

            np.save(
                os.path.join(OUT_TARGETS, target_name),
                target.astype(np.float32)
            )

            print(
                f"Par {contador_global:06d} | "
                f"corte {i+1}/{n} | "
                f"orientación global: {mejor_orientacion_global} | "
                f"shift: {shift_xy} | "
                f"SSIM: {score_ssim:.4f}"
            )

            contador_global += 1

        except Exception as e:
            print(f"Error en {paciente}, corte {i}: {e}")

    return contador_global


# ======================================================
# MAIN
# ======================================================

def main():
    os.makedirs(OUT_INPUTS, exist_ok=True)
    os.makedirs(OUT_TARGETS, exist_ok=True)

    pacientes = [
        p for p in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, p))
    ]

    pacientes = sorted(pacientes)

    if not pacientes:
        print("No encontré pacientes en data/raw")
        return

    contador = 0

    for paciente in pacientes:
        contador = procesar_paciente(paciente, contador)

    print("\n" + "=" * 70)
    print("Preprocesamiento terminado.")
    print(f"Total de pares generados: {contador}")
    print(f"Inputs guardados en: {OUT_INPUTS}")
    print(f"Targets guardados en: {OUT_TARGETS}")


if __name__ == "__main__":
    main()