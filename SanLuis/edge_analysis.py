# -*- coding: utf-8 -*-
"""
Utilidades para detección y análisis de bordes en ROIs de imágenes de microscopía.

- Limpia la lógica de la celda del notebook moviéndola a funciones testeables.
- Maneja rutas de forma robusta (pathlib), validaciones y logging.
- Calcula distancias y ángulos entre dos bordes mediante regresión lineal.

Uso típico en el notebook (en SanLuis/DataProcesing.ipynb):

from edge_analysis import process_rois
results = process_rois(mag="50X")
display(results)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties as un
from scipy import stats as st

__all__ = [
    "ROI",
    "parse_roi",
    "detect_edges",
    "analyze_edges",
    "process_rois",
]


# Configuración básica de logging (silencioso por defecto en librería)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int


def parse_roi(roi_row: pd.Series) -> ROI:
    """Parsea una fila de ROI a un objeto ROI, validando claves y casteando a int.
    Se esperan columnas: 'X', 'Y', 'Width', 'Height'.
    """
    keys = {k.lower(): k for k in roi_row.index}
    try:
        x = int(roi_row[keys.get("x", "X")])
        y = int(roi_row[keys.get("y", "Y")])
        w = int(roi_row[keys.get("width", "Width")])
        h = int(roi_row[keys.get("height", "Height")])
    except KeyError as e:
        raise KeyError(f"Falta columna en ROI CSV: {e}") from e
    return ROI(x=x, y=y, w=w, h=h)


def _load_image_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return img


def _ensure_bgr(img_gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


def draw_rois_on_image(
    img_gray: np.ndarray,
    rois: List[ROI],
    *,
    rect_color: Tuple[int, int, int] = (0, 255, 0),
    rect_thickness: int = 6,
    draw_labels: bool = True,
    label_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Devuelve una imagen BGR de la imagen completa con todos los ROIs dibujados.

    - Dibuja rectángulos de ROIs y opcionalmente índices de ROI.
    - Mantiene la imagen de fondo en escala de grises convertida a BGR.
    """
    overlay = _ensure_bgr(img_gray).copy()
    for idx, roi in enumerate(rois, start=1):
        pt1 = (int(roi.x), int(roi.y))
        pt2 = (int(roi.x + roi.w), int(roi.y + roi.h))
        cv2.rectangle(overlay, pt1, pt2, rect_color, rect_thickness)
        if draw_labels:
            # Posicionar el índice cerca de la esquina superior izquierda del ROI
            text_org = (int(roi.x) + 3, int(roi.y) + 15)
            cv2.putText(
                overlay,
                f"{idx}",
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                label_color,
                1,
                cv2.LINE_AA,
            )
    return overlay


def detect_edges(
    roi: ROI,
    img_gray: np.ndarray,
    img_bgr: Optional[np.ndarray] = None,
    *,
    blur_kernel: Tuple[int, int] = (5, 5),
    canny_low: int = 100,
    canny_high: int = 131,
    rect_color: Tuple[int, int, int] = (0, 255, 0),
    rect_thickness: int = 5,
) -> Tuple[np.ndarray, np.ndarray, ROI]:
    """Detecta bordes (Canny) dentro del ROI y dibuja el ROI + bordes sobre una copia BGR.

    Retorna (edges, overlay_bgr, roi).
    """
    if img_bgr is None:
        img_bgr = _ensure_bgr(img_gray)

    yslice = slice(roi.y, roi.y + roi.h)
    xslice = slice(roi.x, roi.x + roi.w)

    subimg = img_gray[yslice, xslice]
    blur = cv2.GaussianBlur(subimg, blur_kernel, 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    overlay = img_bgr.copy()
    # Pegamos los bordes sobre el canal rojo de la imagen original
    red_channel = overlay[yslice, xslice, 2]
    overlay[yslice, xslice, 2] = np.maximum(red_channel, edges)
    # Rectángulo del ROI
    cv2.rectangle(overlay, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), rect_color, rect_thickness)

    return edges, overlay, roi

def _fit_two_edges_along_axis(edges: np.ndarray, orientation_token: str):
    
    """Extrae dos contornos (superior/inferior o izquierdo/derecho) y ajusta rectas.

    - Si 'Y' está en el nombre de zona, se recorre por columnas (y = m*x + b).
    - En caso contrario, se recorre por filas (x = m*y + b).

    Retorna: (distances[np.ndarray], ang1_deg[float], ang2_deg[float], m1[float], m2[float])
    o None si no hay suficientes puntos para la regresión.
    """
    hh, ww = edges.shape
    if "Y" in orientation_token:
        x_coords: List[int] = []
        y1: List[float] = []
        y2: List[float] = []
        
        for col in range(ww):
            col_vec = edges[:, col]
            xs = np.flatnonzero(col_vec)
            
            if xs.size >= 2:
                x_coords.append(col)
                y1.append(float(xs[-1]))  # borde "inferior"
                y2.append(float(xs[0]))   # borde "superior"
                
        if len(x_coords) < 2:
            return None
        
        m1, b1, _, _, _ = st.linregress(x_coords, y1)
        m2, b2, _, _, _ = st.linregress(x_coords, y2)
        
        distances = np.abs(np.array(y1) - np.array(y2))
        
        # y = m*x + b  => ángulo con eje x: atan(m)
        ang1 = math.degrees(math.atan2(m1, 1.0))
        ang2 = math.degrees(math.atan2(m2, 1.0))
        return distances, ang1, ang2, m1, m2
    else:
        y_coords: List[int] = []
        x1: List[float] = []
        x2: List[float] = []
        
        for row in range(hh):
            ys = np.flatnonzero(edges[row, :])
            
            if ys.size >= 2:
                y_coords.append(row)
                x1.append(float(ys[-1]))  # borde derecho
                x2.append(float(ys[0]))   # borde izquierdo
                
        if len(y_coords) < 2:
            return None
        
        m1, b1, _, _, _ = st.linregress(y_coords, x1)
        m2, b2, _, _, _ = st.linregress(y_coords, x2)
        
        distances = np.abs(np.array(x1) - np.array(x2))
        # x = m*y + b  => tan(theta) = dy/dx = 1/m
        
        ang1 = 90.0 if m1 == 0 else math.degrees(math.atan2(1.0, m1))
        ang2 = 90.0 if m2 == 0 else math.degrees(math.atan2(1.0, m2))
        return distances, ang1, ang2, m1, m2


essential_cols = ["roi","X_centro","Y_centro", "zona", "efecto", "distance_mean", "distance_std", "distance_ufloat", "angle1_deg", "angle2_deg", "delta_deg"]


def analyze_edges(edges: np.ndarray, *, zona: str) -> Optional[Dict[str, float]]:
    """Calcula métricas de distancias y ángulos entre dos bordes detectados.
    Devuelve None si no hay suficientes puntos para regresión.
    """
    res = _fit_two_edges_along_axis(edges, orientation_token=zona)
    if res is None:
        return None
    distances, ang1, ang2, m1, m2 = res
    if distances.size == 0:
        return None

    delta = abs(ang2 - ang1)
    if delta > 90:
        delta = abs(delta - 180)

    mean = float(np.mean(distances))

    std = float(np.std(distances, ddof=1)) if distances.size > 1 else 0.1
    if std <0.5:
        std = 0.5
    uval = un.ufloat(mean, std)

    return {
        "distance_mean": mean,
        "distance_std": std,
        "distance_ufloat": uval,
        "angle1_deg": float(ang1),
        "angle2_deg": float(ang2),
        "delta_deg": float(delta),
    }


def process_rois(
    mag: str = "50X",
    parte="A2",
    imagenes: Optional[List[str]] = None,
    zonas: Optional[List[str]] = None,
    base: Optional[Path] = None,
    *,
    show: bool = True,
    save_dir: Optional[Path] = None,
    no_margins: bool = False,
    show_overview: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta la detección y análisis de bordes para cada ROI y cada efecto de imagen.

    - mag: magnificación, e.g., '50X'
    - imagenes: lista de sufijos de archivo, por defecto ['', '_binario', '_sombra', '_sombra_binario']
    - zonas: lista de sufijos de CSV, por defecto ['_Cuadrado_X', '_Cuadrado_Y', '_Pelos_X']
    - base: ruta base donde están los archivos de Pasantia SL/Mediciones. Si no se indica,
            se usa ../Pasantia SL/Mediciones relativo a este archivo.
    - show: si True, muestra figuras con plt.show()
    - save_dir: si se provee, guarda las figuras de cada ROI en esa carpeta.
    - no_margins: elimina márgenes/espacios en blanco (sin suptitle ni títulos) y ajusta subplots a ocupar todo el lienzo.

    Retorna un DataFrame con resultados por ROI/zona/efecto.
    """
    logger.debug("Iniciando process_rois")

    if imagenes is None:
        imagenes = ["_original", "_binario", "_sombra", "_sombra_binario"]
    if zonas is None:
        zonas = ["_Cuadrado_X", "_Cuadrado_Y", "_Pelos_X"]

    if base is None:
        # ../Pasantia SL/Mediciones relativo a SanLuis/
        base = Path(__file__).resolve().parent.parent / "Pasantia SL" / "Mediciones"
    base = Path(base)

    if not base.exists():
        logger.warning(f"La carpeta base no existe: {base}")

    results: List[Dict[str, object]] = []

    for zona in zonas:
        roi_csv = base / f"{parte}_{mag}{zona}.csv"
        if not roi_csv.exists():
            logger.warning(f"No existe ROI CSV: {roi_csv}")
            continue

        rois_df = pd.read_csv(roi_csv)

        # Parsear todos los ROIs de la zona por adelantado
        rois_list: List[ROI] = [parse_roi(row) for _, row in rois_df.iterrows()]

        # Generar una imagen "overview" por efecto con la imagen completa y todos los ROIs
        for efecto in imagenes:
            img_path = base / f"{parte}_{mag}{efecto}.tiff"
            try:
                img_gray = _load_image_gray(img_path)
            except FileNotFoundError as e:
                logger.warning(str(e))
                continue

            overview_img = draw_rois_on_image(img_gray, rois_list)
            fig_ov, ax_ov = plt.subplots(1, 1, figsize=(6, 6))
            ax_ov.imshow(cv2.cvtColor(overview_img, cv2.COLOR_BGR2RGB))
            if not no_margins:
                ax_ov.set_title(
                    f"ROIs usados | Imagen:{parte} {mag} {zona.replace('_', ' ')} | {efecto.replace('_', ' ').capitalize()}"
                )
            ax_ov.axis("off")

            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                efecto_name = (efecto.lstrip("_") or "original")
                out = save_dir / f"overview_{mag}{zona}_{efecto_name}.png"
                if no_margins:
                    fig_ov.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0)
                else:
                    fig_ov.savefig(out, dpi=150)
            if show_overview:
                plt.show()
            else:
                plt.close(fig_ov)

        for i, roi in enumerate(rois_list):

            fig, axes = plt.subplots(1, len(imagenes), figsize=(4 * len(imagenes), 4))
            if len(imagenes) == 1:
                axes = [axes]  # normalizar a lista
            if not no_margins:
                fig.suptitle(f"ROI {i + 1} | Imagen:{parte} {mag} {zona.replace('_', ' ')}", fontsize=14)

            for j, efecto in enumerate(imagenes):
                img_path = base / f"{parte}_{mag}{efecto}.tiff"
                try:
                    img_gray = _load_image_gray(img_path)
                except FileNotFoundError as e:
                    logger.warning(str(e))
                    axes[j].set_axis_off()
                    continue

                img_bgr = _ensure_bgr(img_gray)
                edges, overlay, _ = detect_edges(roi, img_gray, img_bgr)

                # Mostrar bordes
                axes[j].imshow(edges, cmap="gray")
                if not no_margins:
                    axes[j].set_title(efecto.replace("_", " ").capitalize())
                axes[j].axis("off")

                # Analizar geometría
                metrics = analyze_edges(edges, zona=zona)
                if metrics is None:
                    logger.info(f"ROI {i + 1} Imagen:{parte} {mag} {efecto or 'original'}: bordes insuficientes para regresión")
                    continue

                results.append(
                    {   "roi":i+1,
                        "X_centro":roi.x+roi.w/2,
                        "Y_centro":roi.y+roi.h/2,
                        "zona": zona,
                        "efecto": efecto or "original",
                        "distance_mean": metrics["distance_mean"],
                        "distance_std": metrics["distance_std"],
                        "distance_ufloat": f"{metrics['distance_ufloat']:.1uf}",
                        "angle1_deg": metrics["angle1_deg"],
                        "angle2_deg": metrics["angle2_deg"],
                        "delta_deg": metrics["delta_deg"],
                    }
                )

            if no_margins:
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            else:
                plt.tight_layout()
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                out = save_dir / f"roi_{i + 1}_{mag}{zona}.png"
                if no_margins:
                    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0)
                else:
                    fig.savefig(out, dpi=150)
            if show:
                plt.show()
            else:
                plt.close(fig)

    df = pd.DataFrame(results, columns=essential_cols)
    return df
