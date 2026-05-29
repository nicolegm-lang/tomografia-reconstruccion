# tomografia-reconstruccion# Estación de Reconstrucción Tomográfica 

Proyecto  para reconstrucción tomográfica a partir de imágenes y sinogramas. La interfaz permite generar sinogramas mediante transformada de Radon, reconstruir imágenes usando retroproyección filtrada, comparar filtros, aplicar realce digital y utilizar una red U-Net residual para autoajuste de reconstrucciones.

## Funciones principales

- Carga de imágenes, DICOM, matrices `.npy`, `.csv` y `.txt`.
- Generación de sinogramas desde imágenes.
- Reconstrucción desde sinogramas externos o generados.
- Soporte para geometría de haz paralelo y aproximación de haz de abanico.
- Comparación de filtros de reconstrucción.
- Realce visual mediante ventana/nivel, gamma, CLAHE, suavizado y nitidez.
- Segmentación y detección de bordes.
- Autoajuste con IA usando una U-Net residual conservadora.

## Instalación

```bash
pip install -r requirements.txt
