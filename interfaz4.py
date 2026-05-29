import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import queue

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from skimage.transform import radon, iradon, resize
from skimage.data import shepp_logan_phantom
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import exposure, filters, feature, morphology
from skimage.filters import unsharp_mask

from scipy.ndimage import gaussian_filter, median_filter, map_coordinates

try:
    import pydicom
except ImportError:
    pydicom = None
    
import torch
import torch.nn as nn
    


# ======================================================
# UTILIDADES
# ======================================================

def normalizar(img):
    img = np.asarray(img, dtype=np.float64)
    img = np.squeeze(img)
    img = np.nan_to_num(img)

    mn = np.min(img)
    mx = np.max(img)

    if mx - mn < 1e-12:
        return np.zeros_like(img)

    return (img - mn) / (mx - mn)


def normalizar_percentil(img, pmin=1, pmax=99):
    img = np.asarray(img, dtype=np.float64)
    img = np.nan_to_num(img)

    a = np.percentile(img, pmin)
    b = np.percentile(img, pmax)

    if b - a < 1e-12:
        return normalizar(img)

    img = np.clip(img, a, b)
    return (img - a) / (b - a)


def crear_phantom(tamano=256):
    img = shepp_logan_phantom()
    img = resize(img, (tamano, tamano), anti_aliasing=True)
    return normalizar(img)


def leer_matriz_texto(ruta):
    try:
        return np.loadtxt(ruta, delimiter=",")
    except Exception:
        return np.loadtxt(ruta)


def leer_dicom(ruta):
    if pydicom is None:
        raise ImportError("Instala pydicom con: pip install pydicom")

    ds = pydicom.dcmread(ruta)
    img = ds.pixel_array.astype(np.float64)

    if img.ndim > 2:
        img = img[0]

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    img = img * slope + intercept

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = np.max(img) - img

    return img


def cargar_como_imagen(ruta, tamano=256):
    ext = os.path.splitext(ruta)[1].lower()

    if ext == ".dcm":
        img = leer_dicom(ruta)
    elif ext == ".npy":
        img = np.load(ruta)
    elif ext in [".csv", ".txt"]:
        img = leer_matriz_texto(ruta)
    else:
        img = imread(ruta)
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = rgb2gray(img)

    img = np.squeeze(img)

    if img.ndim != 2:
        raise ValueError("La imagen debe ser una matriz 2D.")

    img = resize(img, (tamano, tamano), anti_aliasing=True)
    return normalizar(img)


def cargar_como_sinograma(ruta):
    ext = os.path.splitext(ruta)[1].lower()

    if ext == ".dcm":
        sino = leer_dicom(ruta)
    elif ext == ".npy":
        sino = np.load(ruta)
    elif ext in [".csv", ".txt"]:
        sino = leer_matriz_texto(ruta)
    else:
        sino = imread(ruta)
        if sino.ndim == 3:
            if sino.shape[2] == 4:
                sino = sino[:, :, :3]
            sino = rgb2gray(sino)

    sino = np.squeeze(sino)

    if sino.ndim not in [2, 3]:
        raise ValueError(f"El sinograma debe ser 2D o stack 3D. Forma recibida: {sino.shape}")

    return normalizar(sino)


def obtener_filtro(nombre):
    if nombre == "Sin filtro":
        return None
    return nombre


def generar_theta(num, rango="180°"):
    grados = 360.0 if rango == "360°" else 180.0
    return np.linspace(0.0, grados, num, endpoint=False)


def agregar_ruido(sino, nivel):
    if nivel <= 0:
        return sino
    return sino + nivel * np.random.normal(0, 1, sino.shape)


def calcular_rmse(ref, rec):
    ref = normalizar(ref)

    if ref.shape != rec.shape:
        rec = resize(rec, ref.shape, anti_aliasing=True)

    rec = normalizar(rec)

    return np.sqrt(np.mean((ref - rec) ** 2))


def calcular_correlacion(ref, rec):
    ref = normalizar(ref)

    if ref.shape != rec.shape:
        rec = resize(rec, ref.shape, anti_aliasing=True)

    rec = normalizar(rec)

    a = ref.flatten()
    b = rec.flatten()

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0

    return np.corrcoef(a, b)[0, 1]


# ======================================================
# SINOGRAMAS Y GEOMETRÍA
# ======================================================

def eje_stack_sinograma(sino):
    if sino.ndim != 3:
        return None
    return int(np.argmin(sino.shape))


def numero_cortes_sinograma(sino):
    if sino is None:
        return 0

    if sino.ndim == 2:
        return 1

    eje = eje_stack_sinograma(sino)
    return sino.shape[eje]


def extraer_sinograma_2d(sino, indice=0):
    if sino.ndim == 2:
        return normalizar(sino)

    eje = eje_stack_sinograma(sino)
    n = sino.shape[eje]
    indice = int(np.clip(indice, 0, n - 1))

    sino_2d = np.take(sino, indice, axis=eje)
    sino_2d = np.squeeze(sino_2d)

    if sino_2d.ndim != 2:
        raise ValueError(f"No se pudo extraer un sinograma 2D. Forma obtenida: {sino_2d.shape}")

    return normalizar(sino_2d)


def preparar_orientacion_sinograma(sino, orientacion="Automático"):
    sino = np.squeeze(sino)

    if sino.ndim != 2:
        raise ValueError(f"El sinograma debe ser 2D. Forma recibida: {sino.shape}")

    if orientacion == "Detector × Ángulos":
        return sino

    if orientacion == "Ángulos × Detector":
        return sino.T

    if sino.shape[0] > sino.shape[1]:
        return sino.T

    return sino


def recortar_sinograma_util(sino, umbral_relativo=0.035, margen=25, min_keep_frac=0.62):
    """
    Recorta zonas laterales del detector que casi no aportan señal.

    La versión anterior era demasiado conservadora y a veces no recortaba nada.
    Esta versión busca la región útil usando el perfil promedio del detector,
    pero evita cortes excesivos manteniendo al menos una fracción mínima del eje.
    """
    sino = np.asarray(sino, dtype=np.float64)

    if sino.ndim != 2:
        return sino

    alto_original = sino.shape[0]

    sino_norm = normalizar_percentil(sino, 0.5, 99.5)
    perfil = np.mean(np.abs(sino_norm), axis=1)
    perfil = gaussian_filter(perfil, sigma=2.0)

    p_bajo = float(np.percentile(perfil, 5))
    p_alto = float(np.percentile(perfil, 98))

    if p_alto - p_bajo < 1e-8:
        return sino

    thr = p_bajo + float(umbral_relativo) * (p_alto - p_bajo)
    indices = np.where(perfil > thr)[0]

    if len(indices) == 0:
        return sino

    i0 = max(0, int(indices[0]) - int(margen))
    i1 = min(alto_original, int(indices[-1]) + int(margen) + 1)

    alto_recortado = i1 - i0

    # Evita que el recorte elimine demasiada información anatómica.
    min_keep = int(np.ceil(float(min_keep_frac) * alto_original))
    if alto_recortado < min_keep:
        centro = int(round((i0 + i1) / 2))
        i0 = max(0, centro - min_keep // 2)
        i1 = min(alto_original, i0 + min_keep)
        i0 = max(0, i1 - min_keep)

    # Si el cambio es prácticamente imperceptible, conserva el original.
    if (i1 - i0) >= 0.985 * alto_original:
        return sino

    return sino[i0:i1, :]


def reducir_360_a_180_promediado(sino):
    sino = np.asarray(sino, dtype=np.float64)
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
    img = np.asarray(img)
    n, m = img.shape

    y, x = np.ogrid[:n, :m]
    cy, cx = n // 2, m // 2
    r = min(cy, cx)

    mascara = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

    out = np.zeros_like(img)
    out[mascara] = img[mascara]

    return out


def rebinning_fanbeam_a_paralelo(sino_fan, d_fuente=2.0, rango="360°"):
    sino_fan = np.asarray(sino_fan, dtype=np.float64)

    if sino_fan.ndim != 2:
        raise ValueError("El sinograma debe ser 2D para aplicar rebinning.")

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

    return normalizar(sino_paralelo)


def reconstruir_desde_sinograma_preparado(
    sino,
    rango="180°",
    filtro="ramp",
    salida=None,
    aplicar_mascara=True,
    realce=True
):
    theta = generar_theta(sino.shape[1], rango)

    rec = iradon(
        sino,
        theta=theta,
        filter_name=obtener_filtro(filtro),
        output_size=salida,
        circle=False
    )

    rec = normalizar_percentil(rec, 0.5, 99.5)

    if realce:
        # Realce visible de posreconstrucción:
        # mejora contraste local y nitidez sin cambiar el sinograma.
        rec = aplicar_ventana_nivel(rec, center=0.50, width=0.92)
        rec = exposure.adjust_gamma(rec, 0.88)
        rec = exposure.equalize_adapthist(rec, clip_limit=0.012)
        rec = gaussian_filter(rec, sigma=0.12)
        rec = unsharp_mask(
            rec,
            radius=1.0,
            amount=0.75,
            preserve_range=True
        )
        rec = normalizar_percentil(rec, 0.5, 99.5)

    if aplicar_mascara:
        rec = aplicar_mascara_circular(rec)

    return rec


def preparar_sinograma_externo_puro(
    sino_base,
    indice,
    orientacion,
    rango,
    geometria,
    d_fuente,
    shift,
    recorte,
    reducir_180,
    ruido
):
    sino = extraer_sinograma_2d(sino_base, indice)

    sino = preparar_orientacion_sinograma(sino, orientacion)

    sino = np.roll(sino, int(shift), axis=0)

    rango_usado = rango

    if geometria == "Haz de abanico":
        sino = rebinning_fanbeam_a_paralelo(
            sino,
            d_fuente=float(d_fuente),
            rango=rango_usado
        )

    if recorte:
        sino = recortar_sinograma_util(sino)

    if reducir_180 and rango_usado == "360°" and sino.shape[1] >= 300:
        sino = reducir_360_a_180_promediado(sino)
        rango_usado = "180°"

    sino = normalizar_percentil(sino, 0.5, 99.7)
    sino = agregar_ruido(sino, float(ruido))

    return sino, rango_usado


def score_calidad_reconstruccion(img):
    img = normalizar_percentil(img, 1, 99)

    gx = np.gradient(img, axis=1)
    gy = np.gradient(img, axis=0)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    nitidez = np.mean(grad)
    contraste = np.std(img)

    n, m = img.shape
    y, x = np.ogrid[:n, :m]
    cy, cx = n // 2, m // 2
    r = min(cy, cx)

    centro = (x - cx) ** 2 + (y - cy) ** 2 <= (0.85 * r) ** 2
    externo = ~centro

    energia_centro = np.mean(img[centro])
    energia_externa = np.mean(img[externo])

    return 2.0 * nitidez + 1.5 * contraste + 0.4 * energia_centro - 0.8 * energia_externa


# ======================================================
# POSTPROCESAMIENTO
# ======================================================

def aplicar_ventana_nivel(img, center=0.5, width=1.0):
    img = normalizar(img)
    width = max(float(width), 1e-3)

    low = center - width / 2
    high = center + width / 2

    img = np.clip(img, low, high)

    return normalizar(img)


def aplicar_gamma(img, gamma=1.0):
    img = normalizar(img)
    gamma = max(float(gamma), 1e-3)
    return exposure.adjust_gamma(img, gamma)


def aplicar_clahe(img, clip_limit=0.03):
    img = normalizar(img)
    clip_limit = max(float(clip_limit), 0.001)
    return exposure.equalize_adapthist(img, clip_limit=clip_limit)


def aplicar_suavizado(img, sigma=0.8, mediana_size=1):
    img = normalizar(img)

    if sigma > 0:
        img = gaussian_filter(img, sigma=float(sigma))

    mediana_size = int(mediana_size)

    if mediana_size > 1:
        if mediana_size % 2 == 0:
            mediana_size += 1
        img = median_filter(img, size=mediana_size)

    return normalizar(img)


def aplicar_nitidez(img, amount=0.5):
    img = normalizar(img)

    if amount <= 0:
        return img

    img = unsharp_mask(
        img,
        radius=1.2,
        amount=float(amount),
        preserve_range=True
    )

    return normalizar(img)


def detectar_bordes_canny_manual(img, sigma=1.2):
    img = normalizar(img)
    sigma = max(float(sigma), 0.1)

    bordes = feature.canny(img, sigma=sigma)

    return bordes.astype(float)


def segmentar_otsu_manual(img, offset=0.0, suavizado=2):
    """
    Segmentación más conservadora para CT/reconstrucciones.
    Evita pintar toda la cabeza usando:
    - ROI circular para ignorar esquinas/fondo.
    - Otsu calculado solo en región útil.
    - Percentil alto como límite mínimo de umbral.
    - Limpieza morfológica más estricta.
    """
    img = normalizar(img)

    n, m = img.shape
    y, x = np.ogrid[:n, :m]
    cy, cx = n // 2, m // 2
    r = 0.94 * min(cy, cx)
    roi = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

    valores = img[roi]
    valores = valores[valores > np.percentile(valores, 3)]

    if valores.size < 20:
        return np.zeros_like(img, dtype=float)

    try:
        umbral_otsu = filters.threshold_otsu(valores)
    except ValueError:
        return np.zeros_like(img, dtype=float)

    # Umbral mínimo por percentil para evitar segmentar casi todo el tejido suave.
    umbral_percentil = np.percentile(valores, 68)
    umbral = max(umbral_otsu, umbral_percentil) + float(offset)
    umbral = np.clip(umbral, 0.0, 1.0)

    mask = (img > umbral) & roi

    # Limpieza más fuerte: quita objetos pequeños y huecos menores.
    min_size = max(120, int(0.0015 * img.size))
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=max(120, int(0.0012 * img.size)))

    suavizado = int(suavizado)

    if suavizado > 0:
        selem = morphology.disk(suavizado)
        mask = morphology.binary_closing(mask, selem)
        mask = morphology.binary_opening(mask, selem)

    return mask.astype(float)


def superponer_segmentacion(img, mascara, alpha=0.30):
    """
    Superposición transparente y menos agresiva.
    La anatomía de fondo permanece visible.
    """
    img = normalizar(img)
    mascara = normalizar(mascara)
    alpha = float(np.clip(alpha, 0.05, 0.80))

    rgb = np.dstack([img, img, img])

    # Color cálido pero con opacidad controlada.
    color = np.zeros_like(rgb)
    color[:, :, 0] = 1.0
    color[:, :, 1] = 0.25
    color[:, :, 2] = 0.10

    mask3 = mascara[:, :, None]
    out = rgb * (1 - alpha * mask3) + color * (alpha * mask3)

    return np.clip(out, 0, 1)


# ======================================================
# SCROLL FRAME
# ======================================================

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, width=350):
        super().__init__(parent)

        self.canvas = tk.Canvas(
            self,
            bg="#0f172a",
            highlightthickness=0,
            width=width
        )

        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview
        )

        self.scrollable_frame = ttk.Frame(self.canvas)

        self.window = self.canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw"
        )

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Configure>", self.ajustar_ancho)
        self.canvas.bind_all("<MouseWheel>", self.scroll_mouse)

    def ajustar_ancho(self, event):
        self.canvas.itemconfig(self.window, width=event.width)

    def scroll_mouse(self, event):
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

def reducir_sinograma_autoajuste(sino, max_det=256, max_ang=180):
    """
    Reduce temporalmente el sinograma para que el autoajuste sea rápido.
    No afecta el sinograma original ni la reconstrucción final.
    """
    sino = np.asarray(sino, dtype=np.float64)

    det, ang = sino.shape

    nuevo_det = min(det, max_det)
    nuevo_ang = min(ang, max_ang)

    if det == nuevo_det and ang == nuevo_ang:
        return sino

    sino_red = resize(
        sino,
        (nuevo_det, nuevo_ang),
        anti_aliasing=True,
        preserve_range=True
    )

    return normalizar(sino_red)

# ======================================================
# MODELO IA: U-NET RESIDUAL CONSERVADORA
# ======================================================

MODEL_IA_PATH = r"C:\Users\Ness\OneDrive\Documents\Escuela\2026-2\Reconstrucción de imagenes\Nueva carpeta\ia_tomografia\models\unet_ct_residual_conservative1.pth"

MODEL_IA_FALLBACKS = [
    MODEL_IA_PATH,
    "models/unet_ct_residual_conservative.pth",
    "models/unet_ct_residual_enhancer.pth",
]


class DoubleConvIA(nn.Module):
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

        self.enc1 = DoubleConvIA(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConvIA(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConvIA(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConvIA(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConvIA(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConvIA(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConvIA(64, 32)

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

        correction = torch.tanh(self.out(d1)) * self.residual_scale

        return torch.clamp(x + correction, 0, 1)


def normalizar_para_ia(img):
    img = np.asarray(img, dtype=np.float32)
    img = np.nan_to_num(img)

    a, b = np.percentile(img, (1, 99))

    if b - a < 1e-8:
        return np.zeros_like(img, dtype=np.float32)

    img = np.clip(img, a, b)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img.astype(np.float32)


def aplicar_modelo_ia_a_imagen(img, model, device):
    """
    Aplica la red IA a una reconstrucción.
    La red fue entrenada con imágenes 256x256.
    """
    img = normalizar_para_ia(img)
    forma_original = img.shape

    img_256 = resize(
        img,
        (256, 256),
        anti_aliasing=True
    ).astype(np.float32)

    x = torch.from_numpy(img_256).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        pred = model(x).cpu().squeeze().numpy()

    pred = normalizar_para_ia(pred)

    if forma_original != (256, 256):
        pred = resize(
            pred,
            forma_original,
            anti_aliasing=True
        )

    pred = normalizar_para_ia(pred)

    return pred.astype(np.float32)


def aplicar_realce_suave_post_ia(img):
    """
    Realce ligero después de la IA:
    contraste, gamma, CLAHE suave, suavizado leve y nitidez moderada.
    """
    img = normalizar_para_ia(img)

    img = np.clip(1.10 * img, 0, 1)
    img = exposure.adjust_gamma(img, 0.95)

    img = exposure.equalize_adapthist(
        img,
        clip_limit=0.010
    )

    img = gaussian_filter(img, sigma=0.15)

    img = unsharp_mask(
        img,
        radius=1.0,
        amount=0.35,
        preserve_range=True
    )

    return np.clip(img, 0, 1).astype(np.float32)

# ======================================================
# APP PRINCIPAL
# ======================================================

class TomografiaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Estación de Reconstrucción Tomográfica")
        self.root.geometry("1650x920")
        self.root.configure(bg="#0f172a")

        # Datos globales
        self.imagen_actual = crear_phantom()
        self.sinograma_externo = None
        self.sinograma_generado = None
        self.sinograma_actual = None
        self.reconstruccion_actual = None
        self.reconstruccion_procesada = None
        self.referencia_metricas = self.imagen_actual.copy()

        self.nombre_imagen = "Phantom de referencia"
        self.nombre_sinograma = "No cargado"

        # Estado de cálculo
        self.busy = False
        
        self.app_alive = True
        self.ui_queue = queue.Queue()
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_app)
        self.root.after(100, self.procesar_cola_ui)

        # Variables generales
        self.num_proy = tk.IntVar(value=180)
        # Ruido separado por módulo para que el comparador no afecte otros flujos
        self.ruido = tk.DoubleVar(value=0.00)  # compatibilidad interna
        self.ruido_imagen_sino = tk.DoubleVar(value=0.00)
        self.ruido_comparador = tk.DoubleVar(value=0.00)
        self.ruido_reconstruccion = tk.DoubleVar(value=0.00)
        self.filtro = tk.StringVar(value="ramp")
        self.cmap = tk.StringVar(value="gray")

        # Sinogramas externos
        self.sino_source = tk.StringVar(value="Sinograma externo")
        self.indice_sinograma = tk.IntVar(value=0)
        self.orientacion_sino = tk.StringVar(value="Automático")
        self.rango_angular = tk.StringVar(value="360°")
        self.geometria_sino = tk.StringVar(value="Haz paralelo")
        self.d_fuente_sino = tk.DoubleVar(value=4.0)
        self.shift_centro = tk.IntVar(value=0)
        self.usar_recorte = tk.BooleanVar(value=False)
        self.usar_mascara = tk.BooleanVar(value=True)
        self.reducir_180 = tk.BooleanVar(value=False)
        self.realce_recon = tk.BooleanVar(value=True)

        # Ajuste y análisis
        self.window_center = tk.DoubleVar(value=0.50)
        self.window_width = tk.DoubleVar(value=1.00)
        self.gamma = tk.DoubleVar(value=1.00)
        self.clahe_clip = tk.DoubleVar(value=0.03)
        self.gauss_sigma = tk.DoubleVar(value=0.80)
        self.mediana_size = tk.IntVar(value=1)
        self.sharpen_amount = tk.DoubleVar(value=0.50)
        self.canny_sigma = tk.DoubleVar(value=1.20)
        self.umbral_offset = tk.DoubleVar(value=0.00)
        self.suavizado_mascara = tk.IntVar(value=2)

        # Elementos de UI
        self.figures = {}
        self.canvases = {}
        self.status_cards = {}
        self.metric_cards = {}

        self.lbl_img_info = None
        self.lbl_sino_info = None
        self.status_text = tk.StringVar(value="Listo")
        self.progress = None

        self.sino_slice_slider = None
        self.sino_slice_label = None
        
        # Modelo IA
        self.modelo_ia = None
        self.dispositivo_ia = "cuda" if torch.cuda.is_available() else "cpu"

        # Resultados de ajuste / autoajuste para visualizar y guardar
        self.modo_ajuste_vista = tk.StringVar(value="Realce manual")
        self.ultimo_resultado_ajuste = {}
        self.guardar_opcion_1 = tk.BooleanVar(value=True)
        self.guardar_opcion_2 = tk.BooleanVar(value=True)
        self.guardar_opcion_3 = tk.BooleanVar(value=False)
        self.guardar_opcion_4 = tk.BooleanVar(value=False)
        self.guardar_label_1 = tk.StringVar(value="Imagen 1")
        self.guardar_label_2 = tk.StringVar(value="Imagen 2")
        self.guardar_label_3 = tk.StringVar(value="Imagen 3")
        self.guardar_label_4 = tk.StringVar(value="Imagen 4")

        self.configurar_estilo()
        self.crear_interfaz()

    # --------------------------------------------------
    # ESTILO
    # --------------------------------------------------

    def configurar_estilo(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background="#0f172a")
        style.configure("Card.TFrame", background="#1e293b")
        style.configure("TNotebook", background="#0f172a", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Segoe UI", 10, "bold"))

        style.map(
            "TNotebook.Tab",
            background=[("selected", "#2563eb"), ("!selected", "#1e293b")],
            foreground=[("selected", "#ffffff"), ("!selected", "#cbd5e1")]
        )

        style.configure(
            "TLabel",
            background="#0f172a",
            foreground="#e2e8f0",
            font=("Segoe UI", 10)
        )

        style.configure(
            "Title.TLabel",
            background="#0f172a",
            foreground="#f8fafc",
            font=("Segoe UI", 18, "bold")
        )

        style.configure(
            "Subtitle.TLabel",
            background="#0f172a",
            foreground="#94a3b8",
            font=("Segoe UI", 10)
        )

        style.configure(
            "CardText.TLabel",
            background="#1e293b",
            foreground="#cbd5e1",
            font=("Segoe UI", 10)
        )

        style.configure(
            "Metric.TLabel",
            background="#1e293b",
            foreground="#ffffff",
            font=("Segoe UI", 11, "bold")
        )

        style.configure(
            "TCombobox",
            padding=5
        )

        style.configure(
            "TCheckbutton",
            background="#0f172a",
            foreground="#e2e8f0",
            font=("Segoe UI", 10)
        )

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#1e293b",
            background="#22c55e",
            bordercolor="#0f172a",
            lightcolor="#22c55e",
            darkcolor="#22c55e"
        )

    # --------------------------------------------------
    # BOTONES BONITOS
    # --------------------------------------------------

    def boton(self, parent, text, command, color="#2563eb", hover="#1d4ed8"):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="white",
            activebackground=hover,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=9,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2"
        )

        btn.bind("<Enter>", lambda e: btn.configure(bg=hover))
        btn.bind("<Leave>", lambda e: btn.configure(bg=color))
        btn.bind("<ButtonPress-1>", lambda e: btn.configure(bg="#0f766e"))
        btn.bind("<ButtonRelease-1>", lambda e: btn.configure(bg=hover))

        btn.pack(fill="x", pady=5)

        return btn

    # --------------------------------------------------
    # INTERFAZ
    # --------------------------------------------------

    def crear_interfaz(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        self.crear_panel_izquierdo(main)
        self.crear_panel_principal(main)

    def crear_panel_izquierdo(self, parent):
        panel = ttk.Frame(parent, width=310)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))

        ttk.Label(
            panel,
            text="Datos de entrada",
            style="Title.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            panel,
            text="Carga imágenes o sinogramas. Todas las pestañas usan los datos activos.",
            style="Subtitle.TLabel",
            wraplength=280
        ).pack(anchor="w", pady=(4, 14))

        self.boton(panel, "Cargar imagen / DICOM", self.cargar_imagen, "#2563eb", "#1d4ed8")
        self.boton(panel, "Cargar sinograma", self.cargar_sinograma, "#06b6d4", "#0891b2")
        self.boton(panel, "Usar phantom de referencia", self.usar_phantom, "#7c3aed", "#6d28d9")

        ttk.Separator(panel).pack(fill="x", pady=14)

        card = ttk.Frame(panel, style="Card.TFrame", padding=12)
        card.pack(fill="x", pady=4)

        ttk.Label(
            card,
            text="Imagen activa",
            background="#1e293b",
            foreground="#ffffff",
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")

        self.lbl_img_info = ttk.Label(
            card,
            text=f"{self.nombre_imagen}\nForma: {self.imagen_actual.shape}",
            style="CardText.TLabel",
            wraplength=260
        )
        self.lbl_img_info.pack(anchor="w", pady=(4, 10))

        ttk.Label(
            card,
            text="Sinograma activo",
            background="#1e293b",
            foreground="#ffffff",
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")

        self.lbl_sino_info = ttk.Label(
            card,
            text=self.nombre_sinograma,
            style="CardText.TLabel",
            wraplength=260
        )
        self.lbl_sino_info.pack(anchor="w", pady=(4, 0))

        ttk.Separator(panel).pack(fill="x", pady=14)

        self.boton(panel, "Actualizar pestaña actual", self.actualizar_pestana_actual, "#22c55e", "#16a34a")
        self.boton(panel, "Guardar sinograma actual", self.guardar_sinograma, "#f97316", "#ea580c")
        self.boton(panel, "Guardar reconstrucción actual", self.guardar_reconstruccion, "#f97316", "#ea580c")

        ttk.Separator(panel).pack(fill="x", pady=14)

        ttk.Label(
            panel,
            textvariable=self.status_text,
            style="Subtitle.TLabel",
            wraplength=280
        ).pack(anchor="w", pady=(0, 6))

        self.progress = ttk.Progressbar(
            panel,
            mode="indeterminate",
            style="Horizontal.TProgressbar"
        )
        self.progress.pack(fill="x")

    def crear_panel_principal(self, parent):
        self.workflow_notebook = ttk.Notebook(parent)
        self.workflow_notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.panel_imagen = ttk.Frame(self.workflow_notebook)
        self.workflow_notebook.add(self.panel_imagen, text="Flujo imagen")

        self.panel_sinograma = ttk.Frame(self.workflow_notebook)
        self.workflow_notebook.add(self.panel_sinograma, text="Flujo sinograma")

        self.notebook_img = ttk.Notebook(self.panel_imagen)
        self.notebook_img.pack(fill=tk.BOTH, expand=True)

        self.notebook_sino = ttk.Notebook(self.panel_sinograma)
        self.notebook_sino.pack(fill=tk.BOTH, expand=True)

        self.crear_tab_sinograma()
        self.crear_tab_comparador()

        self.crear_tab_reconstruccion()
        self.crear_tab_ajuste()
        self.crear_tab_visor()

        # Cambiar de flujo no debe recalcular ni alterar el estado del otro flujo.
        # Solo las pestañas internas actualizan su propio contenido.
        self.workflow_notebook.bind("<<NotebookTabChanged>>", lambda e: self.actualizar_estado_flujo())
        self.notebook_img.bind("<<NotebookTabChanged>>", lambda e: self.actualizar_pestana_actual())
        self.notebook_sino.bind("<<NotebookTabChanged>>", lambda e: self.actualizar_pestana_actual())

    # --------------------------------------------------
    # TABS BASE
    # --------------------------------------------------

    def crear_tab_base(self, notebook, titulo):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=titulo)

        controls = ScrollableFrame(tab, width=360)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        plot_frame = ttk.Frame(tab)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        fig = Figure(figsize=(11, 6), dpi=100)
        fig.patch.set_facecolor("#0f172a")

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.figures[titulo] = fig
        self.canvases[titulo] = canvas

        return controls.scrollable_frame, fig, canvas

    def preparar_ejes(self, fig, ncols):
        fig.clear()
        ejes = fig.subplots(1, ncols)

        if not isinstance(ejes, np.ndarray):
            ejes = np.array([ejes])

        ejes = list(ejes.ravel())

        for ax in ejes:
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="#cbd5e1")

            for spine in ax.spines.values():
                spine.set_color("#334155")

            ax.title.set_color("#f8fafc")
            ax.xaxis.label.set_color("#cbd5e1")
            ax.yaxis.label.set_color("#cbd5e1")

        return ejes

    def add_slider(self, parent, label, variable, minimo, maximo, fmt="{:.2f}"):
        fila = ttk.Frame(parent)
        fila.pack(fill="x", pady=(8, 0))

        ttk.Label(fila, text=label).pack(side=tk.LEFT)

        valor = ttk.Label(fila, text=fmt.format(variable.get()))
        valor.pack(side=tk.RIGHT)

        slider = ttk.Scale(
            parent,
            from_=minimo,
            to=maximo,
            orient=tk.HORIZONTAL,
            variable=variable
        )
        slider.pack(fill="x")

        def update(v):
            if isinstance(variable, tk.IntVar):
                variable.set(int(round(float(v))))
            valor.config(text=fmt.format(variable.get()))

        slider.configure(command=update)

        return slider

    def add_status_card(self, parent, key, metricas=False):
        ttk.Separator(parent).pack(fill="x", pady=14)

        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.pack(fill="x", pady=4)

        if metricas:
            rmse = ttk.Label(card, text="RMSE: ---", style="Metric.TLabel")
            corr = ttk.Label(card, text="Correlación: ---", style="Metric.TLabel")
            rmse.pack(anchor="w")
            corr.pack(anchor="w", pady=(4, 8))
            self.metric_cards[key] = (rmse, corr)

        lbl = ttk.Label(
            card,
            text="",
            style="CardText.TLabel",
            wraplength=300,
            justify="left"
        )
        lbl.pack(anchor="w")

        self.status_cards[key] = lbl

    # --------------------------------------------------
    # EJECUCIÓN EN SEGUNDO PLANO
    # --------------------------------------------------

    def run_async(self, mensaje, worker, on_success):
        if self.busy:
            messagebox.showinfo("Proceso en curso", "Espera a que termine el proceso actual.")
            return

        self.busy = True
        self.status_text.set(mensaje)
        self.progress.start(12)
        self.root.configure(cursor="watch")

        def target():
            try:
                result = worker()
                error = None
            except Exception as e:
                result = None
                error = e

            # El thread NO toca la interfaz.
            # Solo manda el resultado a la cola.
            self.ui_queue.put((result, error, on_success))

        threading.Thread(target=target, daemon=True).start()

    def procesar_cola_ui(self):
        """
        Revisa resultados de procesos en segundo plano.
        Esta función sí corre en el hilo principal de Tkinter.
        """
        if not self.app_alive:
            return

        try:
            while True:
                result, error, on_success = self.ui_queue.get_nowait()

                self.progress.stop()
                self.root.configure(cursor="")
                self.busy = False

                if error is not None:
                    self.status_text.set("Error")
                    messagebox.showerror("Error", str(error))
                else:
                    on_success(result)
                    self.status_text.set("Listo")

        except queue.Empty:
            pass

        self.root.after(100, self.procesar_cola_ui)


    def cerrar_app(self):
        """
        Cierra la aplicación evitando que un thread intente actualizar
        la interfaz después de cerrar la ventana.
        """
        self.app_alive = False

        try:
            self.progress.stop()
        except Exception:
            pass

        self.root.destroy()
        

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------

    def crear_tab_sinograma(self):
        key = "Imagen → Sinograma"
        controls, fig, canvas = self.crear_tab_base(self.notebook_img, key)

        ttk.Label(controls, text="Imagen → Sinograma", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            controls,
            text="Genera el sinograma de la imagen activa mediante la transformada de Radon.",
            style="Subtitle.TLabel",
            wraplength=310
        ).pack(anchor="w", pady=(2, 10))

        self.add_slider(controls, "Número de proyecciones", self.num_proy, 8, 360, "{:.0f}")
        self.add_slider(controls, "Ruido simulado", self.ruido, 0.0, 0.5, "{:.2f}")

        ttk.Label(controls, text="Paleta de visualización").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.cmap,
            values=["gray", "magma", "viridis", "plasma", "inferno", "hot", "turbo"],
            state="readonly"
        ).pack(fill="x", pady=(2, 8))

        self.boton(controls, "Generar sinograma", self.procesar_imagen_a_sinograma, "#06b6d4", "#0891b2")

        self.add_status_card(controls, key, metricas=False)

        self.procesar_imagen_a_sinograma()

    def crear_tab_reconstruccion(self):
        key = "Sinograma → Reconstrucción"
        controls, fig, canvas = self.crear_tab_base(self.notebook_sino, key)

        ttk.Label(controls, text="Sinograma → Reconstrucción", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            controls,
            text="Reconstrucción desde sinograma externo o generado. Incluye geometría, filtros y centro de rotación.",
            style="Subtitle.TLabel",
            wraplength=310
        ).pack(anchor="w", pady=(2, 10))

        ttk.Label(controls, text="Fuente de sinograma").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.sino_source,
            values=["Sinograma externo", "Sinograma generado"],
            state="readonly"
        ).pack(fill="x", pady=(2, 8))

        ttk.Label(controls, text="Orientación del sinograma").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.orientacion_sino,
            values=["Automático", "Detector × Ángulos", "Ángulos × Detector"],
            state="readonly"
        ).pack(fill="x", pady=(2, 8))

        ttk.Label(controls, text="Rango angular").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.rango_angular,
            values=["180°", "360°"],
            state="readonly"
        ).pack(fill="x", pady=(2, 8))

        ttk.Label(controls, text="Geometría").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.geometria_sino,
            values=["Haz paralelo", "Haz de abanico"],
            state="readonly"
        ).pack(fill="x", pady=(2, 8))

        self.add_slider(controls, "Distancia fuente-centro D", self.d_fuente_sino, 1.2, 6.0, "{:.2f}")

        ttk.Label(controls, text="Corte del stack").pack(anchor="w", pady=(8, 0))
        self.sino_slice_slider = ttk.Scale(
            controls,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.indice_sinograma
        )
        self.sino_slice_slider.pack(fill="x")

        self.sino_slice_label = ttk.Label(controls, text="Sinograma 2D")
        self.sino_slice_label.pack(anchor="center", pady=(2, 8))
        self.sino_slice_slider.state(["disabled"])

        self.add_slider(controls, "Centro de rotación", self.shift_centro, -100, 100, "{:.0f}")


        ttk.Checkbutton(
            controls,
            text="Recortar región útil del detector",
            variable=self.usar_recorte,
            command=self.procesar_sinograma_a_reconstruccion
        ).pack(anchor="w", pady=4)

        ttk.Checkbutton(
            controls,
            text="Reducir 360° a 180° promediado",
            variable=self.reducir_180,
            command=self.procesar_sinograma_a_reconstruccion
        ).pack(anchor="w", pady=4)

        ttk.Checkbutton(
            controls,
            text="Aplicar máscara circular",
            variable=self.usar_mascara,
            command=self.procesar_sinograma_a_reconstruccion
        ).pack(anchor="w", pady=4)

        ttk.Checkbutton(
            controls,
            text="Realce visual",
            variable=self.realce_recon,
            command=self.procesar_sinograma_a_reconstruccion
        ).pack(anchor="w", pady=4)

        ttk.Label(controls, text="Filtro de reconstrucción").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.filtro,
            values=["ramp", "shepp-logan", "cosine", "hamming", "hann", "Sin filtro"],
            state="readonly"
        ).pack(fill="x", pady=(2, 8))

        ttk.Label(controls, text="Paleta de visualización").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.cmap,
            values=["gray", "magma", "viridis", "plasma", "inferno", "hot", "turbo"],
            state="readonly"
        ).pack(fill="x", pady=(2, 8))

        self.boton(controls, "Reconstruir", self.procesar_sinograma_a_reconstruccion, "#22c55e", "#16a34a")

        self.add_status_card(controls, key, metricas=False)

        self.procesar_sinograma_a_reconstruccion()

    def crear_tab_ajuste(self):
        key = "Ajuste y análisis"
        controls, fig, canvas = self.crear_tab_base(self.notebook_sino, key)

        ttk.Label(controls, text="Ajuste y análisis", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            controls,
            text="Ajuste manual de la reconstrucción actual: contraste, suavizado, nitidez, bordes y segmentación.",
            style="Subtitle.TLabel",
            wraplength=310
        ).pack(anchor="w", pady=(2, 10))

        ttk.Label(controls, text="Modo de resultado").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.modo_ajuste_vista,
            values=["Realce manual", "Autoajuste IA"],
            state="readonly"
        ).pack(fill="x", pady=(2, 8))

        self.boton(controls, "Autoajuste visual sugerido", self.aplicar_ajuste_recomendado, "#7c3aed", "#6d28d9")
        self.boton(controls, "Autoajuste IA", self.procesar_autoajuste_ia, "#9333ea", "#7e22ce")
        self.boton(controls, "Mostrar resultado seleccionado", self.renderizar_ajuste_actual, "#0ea5e9", "#0284c7")

        ttk.Separator(controls).pack(fill="x", pady=10)

        self.add_slider(controls, "Nivel", self.window_center, 0.0, 1.0, "{:.2f}")
        self.add_slider(controls, "Ventana", self.window_width, 0.05, 2.0, "{:.2f}")
        self.add_slider(controls, "Gamma", self.gamma, 0.20, 3.00, "{:.2f}")
        self.add_slider(controls, "CLAHE", self.clahe_clip, 0.001, 0.10, "{:.3f}")

        ttk.Separator(controls).pack(fill="x", pady=10)

        self.add_slider(controls, "Suavizado gaussiano", self.gauss_sigma, 0.0, 4.0, "{:.2f}")
        self.add_slider(controls, "Filtro mediana", self.mediana_size, 1, 9, "{:.0f}")
        self.add_slider(controls, "Realce de nitidez", self.sharpen_amount, 0.0, 3.0, "{:.2f}")

        ttk.Separator(controls).pack(fill="x", pady=10)

        self.add_slider(controls, "Suavizado de bordes", self.canny_sigma, 0.2, 5.0, "{:.2f}")
        self.add_slider(controls, "Ajuste de umbral", self.umbral_offset, -0.30, 0.30, "{:.2f}")
        self.add_slider(controls, "Suavizado de máscara", self.suavizado_mascara, 0, 8, "{:.0f}")

        self.boton(controls, "Aplicar ajustes", self.procesar_ajuste_analisis, "#22c55e", "#16a34a")

        ttk.Separator(controls).pack(fill="x", pady=10)
        ttk.Label(controls, text="Guardar imágenes del resultado").pack(anchor="w", pady=(6, 2))
        ttk.Checkbutton(controls, textvariable=self.guardar_label_1, variable=self.guardar_opcion_1).pack(anchor="w")
        ttk.Checkbutton(controls, textvariable=self.guardar_label_2, variable=self.guardar_opcion_2).pack(anchor="w")
        ttk.Checkbutton(controls, textvariable=self.guardar_label_3, variable=self.guardar_opcion_3).pack(anchor="w")
        ttk.Checkbutton(controls, textvariable=self.guardar_label_4, variable=self.guardar_opcion_4).pack(anchor="w")
        self.boton(controls, "Guardar seleccionadas", self.guardar_resultados_ajuste_seleccionados, "#f97316", "#ea580c")

        self.add_status_card(controls, key, metricas=True)

        self.procesar_ajuste_analisis()

    def crear_tab_comparador(self):
        key = "Comparador de filtros"
        controls, fig, canvas = self.crear_tab_base(self.notebook_img, key)

        ttk.Label(controls, text="Comparador de filtros", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            controls,
            text="Compara reconstrucción sin filtro y filtros FBP.",
            style="Subtitle.TLabel",
            wraplength=310
        ).pack(anchor="w", pady=(2, 10))

        self.add_slider(controls, "Número de proyecciones", self.num_proy, 8, 360, "{:.0f}")
        self.add_slider(controls, "Ruido simulado", self.ruido, 0.0, 0.5, "{:.2f}")

        self.boton(controls, "Comparar filtros", self.procesar_comparador_filtros, "#06b6d4", "#0891b2")

        self.add_status_card(controls, key, metricas=False)

        self.procesar_comparador_filtros()

    def crear_tab_visor(self):
        key = "Visor clínico"
        controls, fig, canvas = self.crear_tab_base(self.notebook_sino, key)

        ttk.Label(controls, text="Visor clínico", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            controls,
            text="Vista limpia tipo visor DICOM para la reconstrucción actual o procesada.",
            style="Subtitle.TLabel",
            wraplength=310
        ).pack(anchor="w", pady=(2, 10))

        self.viewer_source = tk.StringVar(value="Reconstrucción procesada")
        self.viewer_wc = tk.DoubleVar(value=0.50)
        self.viewer_ww = tk.DoubleVar(value=1.00)
        self.viewer_gamma = tk.DoubleVar(value=1.00)

        ttk.Label(controls, text="Fuente").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.viewer_source,
            values=["Imagen activa", "Reconstrucción actual", "Reconstrucción procesada"],
            state="readonly"
        ).pack(fill="x", pady=(2, 8))

        self.add_slider(controls, "Centro de ventana", self.viewer_wc, 0.0, 1.0, "{:.2f}")
        self.add_slider(controls, "Ancho de ventana", self.viewer_ww, 0.05, 2.0, "{:.2f}")
        self.add_slider(controls, "Gamma", self.viewer_gamma, 0.2, 3.0, "{:.2f}")

        self.boton(controls, "Actualizar visor", self.procesar_visor_clinico, "#7c3aed", "#6d28d9")

        self.add_status_card(controls, key, metricas=False)

        self.procesar_visor_clinico()

    # --------------------------------------------------
    # CARGA Y GUARDADO
    # --------------------------------------------------

    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen o DICOM",
            filetypes=[
                ("Formatos soportados", "*.png *.jpg *.jpeg *.tif *.tiff *.dcm *.npy *.csv *.txt"),
                ("Imágenes", "*.png *.jpg *.jpeg *.tif *.tiff"),
                ("DICOM", "*.dcm"),
                ("Matrices", "*.npy *.csv *.txt"),
                ("Todos", "*.*")
            ]
        )

        if not ruta:
            return

        def worker():
            return cargar_como_imagen(ruta)

        def done(img):
            self.imagen_actual = img
            self.referencia_metricas = img.copy()
            self.nombre_imagen = os.path.basename(ruta)

            self.lbl_img_info.config(
                text=f"{self.nombre_imagen}\nForma: {self.imagen_actual.shape}"
            )

            self.actualizar_pestana_actual()

        self.run_async("Cargando imagen...", worker, done)

    def cargar_sinograma(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar sinograma",
            filetypes=[
                ("Formatos soportados", "*.png *.jpg *.jpeg *.tif *.tiff *.dcm *.npy *.csv *.txt"),
                ("Imágenes", "*.png *.jpg *.jpeg *.tif *.tiff"),
                ("DICOM", "*.dcm"),
                ("Matrices", "*.npy *.csv *.txt"),
                ("Todos", "*.*")
            ]
        )

        if not ruta:
            return

        def worker():
            return cargar_como_sinograma(ruta)

        def done(sino):
            self.sinograma_externo = sino
            self.nombre_sinograma = os.path.basename(ruta)
            self.sino_source.set("Sinograma externo")

            self.lbl_sino_info.config(
                text=f"{self.nombre_sinograma}\nForma: {self.sinograma_externo.shape}"
            )

            self.actualizar_slider_cortes()
            if hasattr(self, "workflow_notebook"):
                self.workflow_notebook.select(self.panel_sinograma)
            if hasattr(self, "notebook_sino"):
                self.notebook_sino.select(0)
            self.procesar_sinograma_a_reconstruccion()

        self.run_async("Cargando sinograma...", worker, done)

    def usar_phantom(self):
        self.imagen_actual = crear_phantom()
        self.referencia_metricas = self.imagen_actual.copy()
        self.nombre_imagen = "Phantom de referencia"

        self.lbl_img_info.config(
            text=f"{self.nombre_imagen}\nForma: {self.imagen_actual.shape}"
        )

        self.actualizar_pestana_actual()

    def guardar_sinograma(self):
        if self.sinograma_actual is None:
            messagebox.showwarning("Aviso", "Primero genera o carga un sinograma.")
            return

        ruta = filedialog.asksaveasfilename(
            title="Guardar sinograma",
            defaultextension=".npy",
            filetypes=[("NumPy", "*.npy"), ("CSV", "*.csv"), ("PNG", "*.png")]
        )

        if not ruta:
            return

        try:
            if ruta.endswith(".npy"):
                np.save(ruta, self.sinograma_actual)
            elif ruta.endswith(".csv"):
                np.savetxt(ruta, self.sinograma_actual, delimiter=",")
            else:
                import matplotlib.pyplot as plt
                plt.imsave(ruta, normalizar(self.sinograma_actual), cmap="gray")

            messagebox.showinfo("Guardado", "Sinograma guardado correctamente.")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar:\n{e}")

    def guardar_reconstruccion(self):
        rec = self.reconstruccion_procesada if self.reconstruccion_procesada is not None else self.reconstruccion_actual

        if rec is None:
            messagebox.showwarning("Aviso", "Primero ejecuta una reconstrucción.")
            return

        ruta = filedialog.asksaveasfilename(
            title="Guardar reconstrucción",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("NumPy", "*.npy"), ("CSV", "*.csv")]
        )

        if not ruta:
            return

        try:
            rec = normalizar(rec)

            if ruta.endswith(".npy"):
                np.save(ruta, rec)
            elif ruta.endswith(".csv"):
                np.savetxt(ruta, rec, delimiter=",")
            else:
                import matplotlib.pyplot as plt
                plt.imsave(ruta, rec, cmap=self.cmap.get())

            messagebox.showinfo("Guardado", "Reconstrucción guardada correctamente.")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar:\n{e}")

    # --------------------------------------------------
    # ACTUALIZACIÓN
    # --------------------------------------------------

    def actualizar_estado_flujo(self):
        """
        Cambiar entre Flujo imagen y Flujo sinograma no debe recalcular nada.
        Esto evita que volver al flujo de imagen modifique la fuente o parámetros
        usados en Sinograma → Reconstrucción.
        """
        if not hasattr(self, "workflow_notebook"):
            return

        flujo_idx = self.workflow_notebook.index(self.workflow_notebook.select())

        if flujo_idx == 0:
            self.status_text.set("Flujo imagen activo")
        else:
            self.status_text.set("Flujo sinograma activo")


    def actualizar_pestana_actual(self):
        if not hasattr(self, "workflow_notebook"):
            return

        flujo_idx = self.workflow_notebook.index(self.workflow_notebook.select())

        if flujo_idx == 0:
            if not hasattr(self, "notebook_img"):
                return

            idx = self.notebook_img.index(self.notebook_img.select())

            if idx == 0:
                self.procesar_imagen_a_sinograma()
            elif idx == 1:
                self.procesar_comparador_filtros()

        elif flujo_idx == 1:
            if not hasattr(self, "notebook_sino"):
                return

            idx = self.notebook_sino.index(self.notebook_sino.select())

            if idx == 0:
                self.procesar_sinograma_a_reconstruccion()
            elif idx == 1:
                self.procesar_ajuste_analisis()
            elif idx == 2:
                self.procesar_visor_clinico()

    def actualizar_slider_cortes(self):
        if self.sinograma_externo is None or self.sino_slice_slider is None:
            return

        n = numero_cortes_sinograma(self.sinograma_externo)

        if n <= 1:
            self.indice_sinograma.set(0)
            self.sino_slice_slider.configure(to=0)
            self.sino_slice_slider.state(["disabled"])
            self.sino_slice_label.config(text="Sinograma 2D")
        else:
            self.indice_sinograma.set(0)
            self.sino_slice_slider.configure(to=n - 1)
            self.sino_slice_slider.state(["!disabled"])
            self.sino_slice_label.config(text=f"Corte 1 de {n}")

            self.sino_slice_slider.configure(
                command=lambda v: self.sino_slice_label.config(
                    text=f"Corte {int(float(v)) + 1} de {n}"
                )
            )

    # --------------------------------------------------
    # PROCESOS
    # --------------------------------------------------

    def procesar_imagen_a_sinograma(self):
        key = "Imagen → Sinograma"

        img = self.imagen_actual.copy()
        num = int(self.num_proy.get())
        ruido = float(self.ruido_imagen_sino.get())

        def worker():
            theta = generar_theta(num, "180°")
            sino = radon(img, theta=theta)
            sino = agregar_ruido(sino, ruido)
            return sino

        def done(sino):
            # Se guarda como sinograma generado, pero NO cambia automáticamente
            # la fuente del flujo Sinograma → Reconstrucción.
            # Así, volver al flujo de imagen no altera el trabajo hecho con un sinograma externo.
            self.sinograma_generado = sino

            # Para guardar desde el panel izquierdo, dejamos este como último sinograma visible.
            self.sinograma_actual = sino

            fig = self.figures[key]
            canvas = self.canvases[key]

            ax1, ax2 = self.preparar_ejes(fig, 2)

            ax1.imshow(img, cmap=self.cmap.get())
            ax1.set_title("Imagen activa f(x,y)")
            ax1.axis("off")

            ax2.imshow(sino, cmap="gray", aspect="auto", extent=(0, 180, 0, sino.shape[0]))
            ax2.set_title("Sinograma R{f}(t,θ)")
            ax2.set_xlabel("Ángulo θ")
            ax2.set_ylabel("Detector t")

            fig.suptitle(
                f"Generación de sinograma | {num} proyecciones",
                color="#f8fafc",
                fontsize=13,
                fontweight="bold"
            )

            fig.tight_layout(pad=2.0)
            canvas.draw()

            self.status_cards[key].config(
                text="Sinograma generado correctamente. Para usarlo en reconstrucción, selecciona 'Sinograma generado' en el flujo de sinograma."
            )

        self.run_async("Generando sinograma...", worker, done)

    def obtener_sinograma_fuente(self):
        if self.sino_source.get() == "Sinograma generado":
            if self.sinograma_generado is None:
                raise ValueError("No hay sinograma generado. Primero usa la pestaña Imagen → Sinograma.")
            return self.sinograma_generado

        if self.sinograma_externo is None:
            raise ValueError("No hay sinograma externo cargado.")

        return self.sinograma_externo

    def preparar_sinograma_desde_interfaz(self):
        sino_base = self.obtener_sinograma_fuente()

        if sino_base.ndim == 3:
            idx = int(self.indice_sinograma.get())
        else:
            idx = 0

        sino, rango_usado = preparar_sinograma_externo_puro(
            sino_base=sino_base,
            indice=idx,
            orientacion=self.orientacion_sino.get(),
            rango=self.rango_angular.get(),
            geometria=self.geometria_sino.get(),
            d_fuente=float(self.d_fuente_sino.get()),
            shift=int(self.shift_centro.get()),
            recorte=self.usar_recorte.get(),
            reducir_180=self.reducir_180.get(),
            ruido=0.0
        )

        return sino, rango_usado

    def procesar_sinograma_a_reconstruccion(self):
        key = "Sinograma → Reconstrucción"

        if self.sino_source.get() == "Sinograma externo" and self.sinograma_externo is None:
            fig = self.figures[key]
            canvas = self.canvases[key]
            ax1, ax2 = self.preparar_ejes(fig, 2)

            ax1.text(
                0.5,
                0.5,
                "Carga un sinograma desde el panel izquierdo",
                ha="center",
                va="center",
                color="#f8fafc",
                fontsize=13
            )
            ax1.axis("off")
            ax2.axis("off")

            fig.tight_layout(pad=2.0)
            canvas.draw()
            self.status_cards[key].config(text="No hay sinograma externo cargado.")
            return

        try:
            sino_base = self.obtener_sinograma_fuente()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if sino_base.ndim == 3:
            idx = int(self.indice_sinograma.get())
        else:
            idx = 0

        params = {
            "orientacion": self.orientacion_sino.get(),
            "rango": self.rango_angular.get(),
            "geometria": self.geometria_sino.get(),
            "d_fuente": float(self.d_fuente_sino.get()),
            "shift": int(self.shift_centro.get()),
            "recorte": bool(self.usar_recorte.get()),
            "reducir_180": bool(self.reducir_180.get()),
            "ruido": float(self.ruido_reconstruccion.get()),
            "filtro": self.filtro.get(),
            "mascara": bool(self.usar_mascara.get()),
            "realce": bool(self.realce_recon.get()),
            "fuente": self.sino_source.get(),
            "total": numero_cortes_sinograma(sino_base),
            "cmap": self.cmap.get(),
        }

        def worker():
            sino, rango_usado = preparar_sinograma_externo_puro(
                sino_base=sino_base,
                indice=idx,
                orientacion=params["orientacion"],
                rango=params["rango"],
                geometria=params["geometria"],
                d_fuente=params["d_fuente"],
                shift=params["shift"],
                recorte=params["recorte"],
                reducir_180=params["reducir_180"],
                ruido=params["ruido"]
            )

            rec = reconstruir_desde_sinograma_preparado(
                sino,
                rango=rango_usado,
                filtro=params["filtro"],
                salida=None,
                aplicar_mascara=params["mascara"],
                realce=params["realce"]
            )
            return sino, rec, rango_usado, params

        def done(result):
            sino, rec, rango_usado, params_done = result

            self.sinograma_actual = sino
            self.reconstruccion_actual = rec
            self.reconstruccion_procesada = None
            self.referencia_metricas = None

            fig = self.figures[key]
            canvas = self.canvases[key]

            ax1, ax2 = self.preparar_ejes(fig, 2)

            ax1.imshow(
                sino,
                cmap="gray",
                aspect="auto",
                extent=(0, float(rango_usado.replace("°", "")), 0, sino.shape[0])
            )
            ax1.set_title("Sinograma preparado")
            ax1.set_xlabel("Ángulo θ")
            ax1.set_ylabel("Detector t")

            ax2.imshow(rec, cmap=params_done["cmap"])
            ax2.set_title("Reconstrucción estimada")
            ax2.axis("off")

            total = params_done["total"]
            corte_txt = f"Corte {idx + 1} de {total}" if total > 1 else "Sinograma 2D"

            fig.suptitle(
                f"{params_done['fuente']} | {corte_txt} | {params_done['geometria']} | Rango: {rango_usado} | Filtro: {params_done['filtro']}",
                color="#f8fafc",
                fontsize=13,
                fontweight="bold"
            )

            fig.tight_layout(pad=2.0)
            canvas.draw()

            self.status_cards[key].config(
                text="Reconstrucción completada. Puedes continuar con ajuste, análisis o visor clínico."
            )

        self.run_async("Reconstruyendo desde sinograma...", worker, done)

    def autoajustar_sinograma(self):
        if self.sino_source.get() == "Sinograma externo" and self.sinograma_externo is None:
            messagebox.showwarning("Aviso", "Primero carga un sinograma.")
            return

        try:
            sino_base = self.obtener_sinograma_fuente()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        idx = int(self.indice_sinograma.get()) if sino_base.ndim == 3 else 0

        def worker():
            sino_2d = extraer_sinograma_2d(sino_base, idx)

            # Autoajuste rápido: menos combinaciones y baja resolución
            orientaciones = [
                "Ángulos × Detector",
                "Detector × Ángulos"
            ]

            rangos = [
                "180°",
                "360°"
            ]

            geometrias = [
                "Haz paralelo",
                "Haz de abanico"
            ]

            filtros = [
                "ramp",
                "shepp-logan"
            ]

            # Menos centros para hacerlo más rápido
            shifts = [
                -40,
                0,
                40,
                80
            ]

            d_values = [
                3.0,
                5.0,
                6.0
            ]

            mejor_score = -np.inf
            mejor = None

            for orientacion in orientaciones:
                for rango in rangos:
                    for geometria in geometrias:
                        for filtro in filtros:
                            for shift in shifts:

                                if geometria == "Haz paralelo":
                                    d_lista = [4.0]
                                else:
                                    d_lista = d_values

                                for d in d_lista:
                                    try:
                                        sino_proc = preparar_orientacion_sinograma(
                                            sino_2d,
                                            orientacion
                                        )

                                        sino_proc = np.roll(
                                            sino_proc,
                                            int(shift),
                                            axis=0
                                        )

                                        if geometria == "Haz de abanico":
                                            sino_proc = rebinning_fanbeam_a_paralelo(
                                                sino_proc,
                                                d_fuente=float(d),
                                                rango=rango
                                            )

                                        sino_proc = recortar_sinograma_util(sino_proc)

                                        # Aquí está la clave: reducir antes de reconstruir
                                        sino_proc = reducir_sinograma_autoajuste(
                                            sino_proc,
                                            max_det=220,
                                            max_ang=160
                                        )

                                        sino_proc = normalizar_percentil(
                                            sino_proc,
                                            0.5,
                                            99.7
                                        )

                                        theta = generar_theta(
                                            sino_proc.shape[1],
                                            rango
                                        )

                                        rec = iradon(
                                            sino_proc,
                                            theta=theta,
                                            filter_name=obtener_filtro(filtro),
                                            output_size=160,
                                            circle=False
                                        )

                                        rec = normalizar_percentil(rec, 1, 99)

                                        rec = gaussian_filter(rec, sigma=0.25)

                                        rec = unsharp_mask(
                                            rec,
                                            radius=1.0,
                                            amount=0.5,
                                            preserve_range=True
                                        )

                                        rec = normalizar_percentil(rec, 1, 99)
                                        rec = aplicar_mascara_circular(rec)

                                        score = score_calidad_reconstruccion(rec)

                                        if score > mejor_score:
                                            mejor_score = score
                                            mejor = {
                                                "orientacion": orientacion,
                                                "rango": rango,
                                                "geometria": geometria,
                                                "filtro": filtro,
                                                "shift": shift,
                                                "d": d,
                                                "score": score
                                            }

                                    except Exception:
                                        continue

            if mejor is None:
                raise RuntimeError("No se pudo encontrar una configuración válida.")

            return mejor

        def done(mejor):
            self.orientacion_sino.set(mejor["orientacion"])
            self.rango_angular.set(mejor["rango"])
            self.geometria_sino.set(mejor["geometria"])
            self.filtro.set(mejor["filtro"])
            self.shift_centro.set(mejor["shift"])
            self.d_fuente_sino.set(mejor["d"])
            self.usar_recorte.set(True)
            self.usar_mascara.set(True)

            messagebox.showinfo(
                "Autoajuste finalizado",
                f"Configuración sugerida:\n\n"
                f"Orientación: {mejor['orientacion']}\n"
                f"Rango: {mejor['rango']}\n"
                f"Geometría: {mejor['geometria']}\n"
                f"D: {mejor['d']}\n"
                f"Centro: {mejor['shift']}\n"
                f"Filtro: {mejor['filtro']}\n"
                f"Score: {mejor['score']:.4f}"
            )

            self.procesar_sinograma_a_reconstruccion()

        self.run_async("Autoajustando reconstrucción rápida...", worker, done)

    def procesar_ajuste_analisis(self):
        key = "Ajuste y análisis"

        if self.reconstruccion_actual is None:
            base = normalizar(self.imagen_actual)
            referencia = self.imagen_actual.copy()
        else:
            base = normalizar(self.reconstruccion_actual)
            referencia = self.referencia_metricas

        params = {
            "wc": float(self.window_center.get()),
            "ww": float(self.window_width.get()),
            "gamma": float(self.gamma.get()),
            "clahe": float(self.clahe_clip.get()),
            "gauss": float(self.gauss_sigma.get()),
            "median": int(self.mediana_size.get()),
            "sharp": float(self.sharpen_amount.get()),
            "canny": float(self.canny_sigma.get()),
            "offset": float(self.umbral_offset.get()),
            "mask_smooth": int(self.suavizado_mascara.get())
        }

        def worker():
            ajustada = aplicar_ventana_nivel(base, params["wc"], params["ww"])
            ajustada = aplicar_gamma(ajustada, params["gamma"])
            ajustada = aplicar_clahe(ajustada, params["clahe"])

            filtrada = aplicar_suavizado(
                ajustada,
                sigma=params["gauss"],
                mediana_size=params["median"]
            )

            procesada = aplicar_nitidez(filtrada, amount=params["sharp"])

            bordes = detectar_bordes_canny_manual(procesada, sigma=params["canny"])

            mascara = segmentar_otsu_manual(
                procesada,
                offset=params["offset"],
                suavizado=params["mask_smooth"]
            )

            overlay = superponer_segmentacion(procesada, mascara, alpha=0.30)

            return procesada, bordes, mascara, overlay

        def done(result):
            procesada, bordes, mascara, overlay = result

            self.reconstruccion_procesada = procesada
            self.modo_ajuste_vista.set("Realce manual")
            self.ultimo_resultado_ajuste["Realce manual"] = {
                "titulo": "Realce clásico: contraste, bordes y segmentación",
                "status": "Ajuste visual aplicado. Puedes guardar una o varias imágenes del resultado.",
                "imagenes": [
                    ("Base", base, self.cmap.get()),
                    ("Realce manual", procesada, self.cmap.get()),
                    ("Bordes detectados", bordes, "gray"),
                    ("Segmentación superpuesta", overlay, None),
                ],
                "procesada": procesada,
                "referencia": referencia,
                "base_metricas": base,
            }

            self.renderizar_ajuste_actual()

        self.run_async("Aplicando ajustes...", worker, done)

    def aplicar_ajuste_recomendado(self):
        # Preset menos agresivo: conserva escala de grises y segmenta de forma más selectiva.
        self.window_center.set(0.50)
        self.window_width.set(0.92)
        self.gamma.set(0.95)
        self.clahe_clip.set(0.015)
        self.gauss_sigma.set(0.65)
        self.mediana_size.set(1)
        self.sharpen_amount.set(0.55)
        self.canny_sigma.set(2.20)
        self.umbral_offset.set(0.08)
        self.suavizado_mascara.set(3)

        self.procesar_ajuste_analisis()

    def cargar_modelo_ia(self):
        """
        Carga el modelo IA una sola vez. Busca primero el nombre configurado
        y luego nombres alternativos comunes.
        """
        if self.modelo_ia is not None:
            return self.modelo_ia

        ruta_modelo = None

        for candidato in MODEL_IA_FALLBACKS:
            if os.path.exists(candidato):
                ruta_modelo = candidato
                break

        if ruta_modelo is None:
            raise FileNotFoundError(
                "No encontré el modelo IA.\n\n"
                "Verifica que exista alguno de estos archivos:\n"
                + "\n".join(MODEL_IA_FALLBACKS)
            )

        model = ResidualUNetConservative(
            residual_scale=0.05
        ).to(self.dispositivo_ia)

        state = torch.load(
            ruta_modelo,
            map_location=self.dispositivo_ia
        )

        model.load_state_dict(state)
        model.eval()

        self.modelo_ia = model

        return self.modelo_ia


    def obtener_imagen_para_autoajuste_ia(self):
        """
        La IA se aplica principalmente sobre la reconstrucción actual.
        Si no hay reconstrucción, usa la reconstrucción procesada o la imagen activa.
        """
        if self.reconstruccion_actual is not None:
            return self.reconstruccion_actual, "Reconstrucción actual"

        if self.reconstruccion_procesada is not None:
            return self.reconstruccion_procesada, "Reconstrucción procesada"

        return self.imagen_actual, "Imagen activa"


    def procesar_autoajuste_ia(self):
        key = "Ajuste y análisis"

        img_base, nombre_fuente = self.obtener_imagen_para_autoajuste_ia()

        def worker():
            model = self.cargar_modelo_ia()

            ia = aplicar_modelo_ia_a_imagen(
                img_base,
                model,
                self.dispositivo_ia
            )

            ia_realzada = aplicar_realce_suave_post_ia(ia)

            return ia, ia_realzada

        def done(result):
            ia, ia_realzada = result

            self.reconstruccion_procesada = ia_realzada
            self.modo_ajuste_vista.set("Autoajuste IA")

            base_norm = normalizar_para_ia(img_base)
            ia_norm = normalizar_para_ia(ia)
            ia_realzada_norm = normalizar_para_ia(ia_realzada)
            diferencia = np.abs(base_norm - ia_realzada_norm)

            self.ultimo_resultado_ajuste["Autoajuste IA"] = {
                "titulo": "Autoajuste inteligente: red residual + realce automático",
                "status": (
                    "Autoajuste IA aplicado. La imagen final quedó guardada como "
                    "'Reconstrucción procesada'. Puedes verla en el visor clínico o guardarla."
                ),
                "imagenes": [
                    (f"Entrada\n{nombre_fuente}", base_norm, "gray"),
                    ("IA residual", ia_norm, "gray"),
                    ("Autoajuste IA", ia_realzada_norm, "gray"),
                    ("|Entrada - IA|", diferencia, "magma"),
                ],
                "procesada": ia_realzada_norm,
                "referencia": None,
                "base_metricas": base_norm,
            }

            self.renderizar_ajuste_actual()

        self.run_async("Aplicando autoajuste IA...", worker, done)


    def actualizar_labels_guardado_ajuste(self, imagenes):
        """Actualiza los textos de los checkboxes de guardado según el modo mostrado."""
        labels = [self.guardar_label_1, self.guardar_label_2, self.guardar_label_3, self.guardar_label_4]
        checks = [self.guardar_opcion_1, self.guardar_opcion_2, self.guardar_opcion_3, self.guardar_opcion_4]

        for i in range(4):
            if i < len(imagenes):
                labels[i].set(imagenes[i][0].replace("\n", " "))
            else:
                labels[i].set(f"Imagen {i + 1}")
                checks[i].set(False)


    def renderizar_ajuste_actual(self):
        """
        Renderiza de forma explícita el modo seleccionado en Ajuste y análisis.
        Así ya no parece que la pestaña cambie 'de la nada'.
        """
        key = "Ajuste y análisis"
        modo = self.modo_ajuste_vista.get()

        if modo not in self.ultimo_resultado_ajuste:
            # Si aún no existe ese resultado, genera el correspondiente.
            if modo == "Autoajuste IA":
                self.procesar_autoajuste_ia()
            else:
                self.procesar_ajuste_analisis()
            return

        datos = self.ultimo_resultado_ajuste[modo]
        imagenes = datos.get("imagenes", [])
        self.actualizar_labels_guardado_ajuste(imagenes)

        fig = self.figures[key]
        canvas = self.canvases[key]
        fig.clear()
        fig.patch.set_facecolor("#0f172a")

        axs = fig.subplots(1, len(imagenes))
        axs = list(np.ravel(axs))

        for ax, (titulo, img, cmap) in zip(axs, imagenes):
            ax.set_facecolor("#0f172a")
            if cmap is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
            ax.set_title(titulo, color="#f8fafc", fontsize=12, fontweight="bold")
            ax.axis("off")

        fig.suptitle(
            datos.get("titulo", modo),
            color="#f8fafc",
            fontsize=14,
            fontweight="bold"
        )

        fig.tight_layout(pad=2.0)
        canvas.draw_idle()

        if key in self.metric_cards:
            rmse_lbl, corr_lbl = self.metric_cards[key]
            referencia = datos.get("referencia")
            base_metricas = datos.get("base_metricas")
            if referencia is not None and base_metricas is not None:
                rmse = calcular_rmse(referencia, base_metricas)
                corr = calcular_correlacion(referencia, base_metricas)
                rmse_lbl.config(text=f"RMSE base: {rmse:.4f}")
                corr_lbl.config(text=f"Correlación base: {corr:.4f}")
            else:
                rmse_lbl.config(text="RMSE: no disponible")
                corr_lbl.config(text="Correlación: no disponible")

        if key in self.status_cards:
            self.status_cards[key].config(text=datos.get("status", "Resultado actualizado."))


    def guardar_resultados_ajuste_seleccionados(self):
        """Guarda una o varias imágenes del resultado actualmente mostrado."""
        modo = self.modo_ajuste_vista.get()

        if modo not in self.ultimo_resultado_ajuste:
            messagebox.showwarning("Aviso", "Primero genera un resultado de ajuste o autoajuste IA.")
            return

        datos = self.ultimo_resultado_ajuste[modo]
        imagenes = datos.get("imagenes", [])
        checks = [
            self.guardar_opcion_1.get(),
            self.guardar_opcion_2.get(),
            self.guardar_opcion_3.get(),
            self.guardar_opcion_4.get(),
        ]

        seleccionadas = [(i, item) for i, item in enumerate(imagenes) if i < len(checks) and checks[i]]

        if not seleccionadas:
            messagebox.showwarning("Aviso", "Selecciona al menos una imagen para guardar.")
            return

        carpeta = filedialog.askdirectory(title="Seleccionar carpeta para guardar imágenes")

        if not carpeta:
            return

        try:
            import matplotlib.pyplot as plt

            modo_slug = modo.lower().replace(" ", "_").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")

            guardadas = []
            for idx, (titulo, img, cmap) in seleccionadas:
                titulo_slug = (
                    titulo.lower()
                    .replace("\n", "_")
                    .replace(" ", "_")
                    .replace("|", "")
                    .replace("-", "_")
                    .replace("/", "_")
                )
                titulo_slug = "".join(c for c in titulo_slug if c.isalnum() or c == "_")
                nombre = f"{modo_slug}_{idx + 1}_{titulo_slug}.png"
                ruta = os.path.join(carpeta, nombre)

                arr = np.asarray(img)
                if arr.ndim == 2:
                    plt.imsave(ruta, normalizar(arr), cmap=cmap if cmap is not None else "gray")
                else:
                    plt.imsave(ruta, np.clip(arr, 0, 1))
                guardadas.append(nombre)

            messagebox.showinfo(
                "Guardado",
                "Imágenes guardadas correctamente:\n\n" + "\n".join(guardadas)
            )

        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron guardar las imágenes:\n{e}")


    def procesar_comparador_filtros(self):
        key = "Comparador de filtros"

        img = self.imagen_actual.copy()
        num = int(self.num_proy.get())
        ruido = float(self.ruido_comparador.get())

        def worker():
            theta = generar_theta(num, "180°")
            sino = radon(img, theta=theta)
            sino = agregar_ruido(sino, ruido)

            filtros = [None, "ramp", "shepp-logan", "cosine"]
            nombres = ["Sin filtro", "ramp", "shepp-logan", "cosine"]

            resultados = []

            for filtro, nombre in zip(filtros, nombres):
                rec = iradon(
                    sino,
                    theta=theta,
                    filter_name=filtro,
                    output_size=img.shape[0]
                )

                rec = normalizar_percentil(rec, 1, 99)

                rmse = calcular_rmse(img, rec)
                corr = calcular_correlacion(img, rec)

                resultados.append((nombre, rec, rmse, corr))

            return sino, resultados

        def done(result):
            sino, resultados = result
            self.sinograma_actual = sino

            fig = self.figures[key]
            canvas = self.canvases[key]

            fig.clear()
            axs = fig.subplots(1, 4)
            axs = list(np.ravel(axs))

            for ax, (nombre, rec, rmse, corr) in zip(axs, resultados):
                ax.set_facecolor("#0f172a")
                ax.imshow(rec, cmap=self.cmap.get())
                ax.set_title(f"{nombre}\nRMSE={rmse:.3f}", color="#f8fafc")
                ax.axis("off")

            fig.suptitle(
                f"Comparador de filtros | {num} proyecciones",
                color="#f8fafc",
                fontsize=13,
                fontweight="bold"
            )

            fig.tight_layout(pad=2.0)
            canvas.draw()

            mejor = min(resultados, key=lambda x: x[2])

            self.status_cards[key].config(
                text=f"Mejor RMSE: {mejor[0]} | RMSE={mejor[2]:.4f} | correlación={mejor[3]:.4f}"
            )

        self.run_async("Comparando filtros...", worker, done)

    def procesar_visor_clinico(self):
        key = "Visor clínico"

        fuente = self.viewer_source.get()

        if fuente == "Reconstrucción procesada" and self.reconstruccion_procesada is not None:
            img = self.reconstruccion_procesada
        elif fuente == "Reconstrucción actual" and self.reconstruccion_actual is not None:
            img = self.reconstruccion_actual
        else:
            img = self.imagen_actual

        img = normalizar(img)

        wc = float(self.viewer_wc.get())
        ww = float(self.viewer_ww.get())
        gamma = float(self.viewer_gamma.get())

        img_view = aplicar_ventana_nivel(img, wc, ww)
        img_view = aplicar_gamma(img_view, gamma)

        fig = self.figures[key]
        canvas = self.canvases[key]

        fig.clear()
        fig.patch.set_facecolor("black")

        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("black")
        ax.imshow(img_view, cmap="gray", interpolation="nearest")
        ax.axis("off")

        ax.text(
            0.01,
            0.98,
            f"Fuente: {fuente}\nReconstrucción tomográfica",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            va="top",
            ha="left"
        )

        ax.text(
            0.99,
            0.98,
            "CT",
            transform=ax.transAxes,
            color="white",
            fontsize=12,
            va="top",
            ha="right"
        )

        ax.text(
            0.01,
            0.02,
            f"RES: {img.shape[0]}x{img.shape[1]}",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            va="bottom",
            ha="left"
        )

        ax.text(
            0.99,
            0.02,
            f"WC: {wc:.2f}\nWW: {ww:.2f}",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            va="bottom",
            ha="right"
        )

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        canvas.draw()

        self.status_cards[key].config(
            text="Vista tipo visor clínico. Ajusta ventana, ancho y gamma para mejorar la visualización."
        )


# ======================================================
# EJECUCIÓN
# ======================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = TomografiaApp(root)
    root.mainloop()