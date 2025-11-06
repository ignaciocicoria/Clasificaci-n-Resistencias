import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import groupby
from operator import itemgetter
from scipy.signal import find_peaks
from pathlib import Path

INPUT_DIR = r"C:\Users\Usuario\PDI1\tp2\TP2_PDI_CICORIA_RICCI\Resistencias"
OUTPUT_DIR = r"C:\Users\Usuario\PDI1\tp2\TP2_PDI_CICORIA_RICCI\Resistencias_out"

DEBUG_DIR = "debug_fallas"

def rectify_resistor(image_path):
    """
    Detecta el rectángulo azul de la resistencia en la imagen y lo “endereza” (rectifica),
      generando una vista frontal  del cuerpo de la resistencia.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo leer la imagen {image_path}")
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 30, 30])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=1.5)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"[FALLO] Sin contornos en {image_path}")
        return None
    max_area = 0
    best_approx = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and area > max_area:
                best_approx = approx
                max_area = area
            elif best_approx is None:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = box.astype(np.intp)
                if area > max_area:
                    best_approx = box
                    max_area = area
    if best_approx is None:
        print(f"[FALLO] Sin rectángulo en {image_path}")
        return None
    pts = best_approx.reshape(4, 2).astype("float32")
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    rect = order_points(pts)
    width = 400
    height = 150
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    H, status = cv2.findHomography(rect, dst)
    warped = cv2.warpPerspective(image, H, (width, height))
    return warped

def preproceso_deteccion(image):
    """
    Recorta el cuerpo de la resistencia dentro de la imagen (elimina extremos y fondos),
      usando máscaras para separar el cuerpo del fondo azul.
    """
    h, w = image.shape[:2]
    margin = 10
    x1_margin = margin
    y1_margin = margin
    x2_margin = w - margin
    y2_margin = h - margin
    if x2_margin <= x1_margin or y2_margin <= y1_margin:
        print("La imagen es demasiado pequeña para recortar con margen de 10 píxeles.")
        return None
    image_cropped = image[y1_margin:y2_margin, x1_margin:x2_margin]
    hsv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 20])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_resistor = cv2.bitwise_not(mask_blue)
    white_bg = np.full_like(image_cropped, 255)
    resistor_only = np.where(mask_resistor[:, :, np.newaxis] == 255, image_cropped, white_bg)
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)); plt.title("Original recortada")
    # plt.subplot(1, 3, 2); plt.imshow(mask_resistor, cmap='gray'); plt.title("Máscara Resistencia")
    # plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(resistor_only, cv2.COLOR_BGR2RGB)); plt.title("Solo Resistencia")
    # plt.tight_layout(); plt.show()
    proj_vertical = np.sum(mask_resistor, axis=0)
    umbral = 5000
    columnas_blancas = np.where(proj_vertical > umbral)[0]
    def mayor_rango_consecutivo(indices):
        groups = []
        for k, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            groups.append(group)
        longest = max(groups, key=len)
        return longest[0], longest[-1]
    if len(columnas_blancas) > 0:
        x1_body, x2_body = mayor_rango_consecutivo(columnas_blancas)
        mask_cuerpo = np.zeros_like(mask_resistor)
        mask_cuerpo[:, x1_body:x2_body] = mask_resistor[:, x1_body:x2_body]
        img_cuerpo = np.where(mask_cuerpo[:, :, np.newaxis] == 255, image_cropped, white_bg).astype(np.uint8)
        filas_con_contenido = np.where(np.any(mask_cuerpo == 255, axis=1))[0]
        if len(filas_con_contenido) == 0:
            print("No se encontraron filas con contenido en el cuerpo.")
            return None
        y1_body, y2_body = filas_con_contenido[0], filas_con_contenido[-1]
        img_cuerpo_recortado = img_cuerpo[y1_body:y2_body, x1_body:x2_body]
        # Ajuste: recortar 10 píxeles arriba y abajo, y 5 píxeles a los costados
        recorte_arriba = 10
        recorte_abajo = 10
        recorte_izquierda = 5
        recorte_derecha = 5
        h_c, w_c = img_cuerpo_recortado.shape[:2]
        y_start = recorte_arriba if h_c > 2 * recorte_arriba else 0
        y_end = h_c - recorte_abajo if h_c > 2 * recorte_abajo else h_c
        x_start = recorte_izquierda if w_c > 2 * recorte_izquierda else 0
        x_end = w_c - recorte_derecha if w_c > 2 * recorte_derecha else w_c
        img_cuerpo_recortado = img_cuerpo_recortado[y_start:y_end, x_start:x_end]
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 3, 1); plt.imshow(mask_resistor, cmap='gray'); plt.title("Máscara original")
        # plt.subplot(1, 3, 2); plt.plot(proj_vertical); plt.axvline(x1_body, color='r'); plt.axvline(x2_body, color='r'); plt.title("Proyección vertical")
        # plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(img_cuerpo_recortado, cv2.COLOR_BGR2RGB)); plt.title("Cuerpo recortado final")
        # plt.tight_layout(); plt.show()
        print(f"Recorte final: columnas {x1_body + x_start}-{x1_body + x_end}, filas {y1_body + y_start}-{y1_body + y_end}")
        return img_cuerpo_recortado

def definir_colores_hsv():
    """
    Define los rangos HSV para los diferentes colores de bandas de resistencias.
    """
    return {
        'Negro': {'hsv_range': [(np.array([0, 0, 0]), np.array([180, 255, 70]))]},
        'Marrón': {'hsv_range': [(np.array([0, 90, 0]), np.array([11, 260, 130]))]},
        'Rojo': {'hsv_range': [(np.array([0, 80, 60]), np.array([12, 255, 255]))]},
        'Naranja': {'hsv_range': [(np.array([10, 120, 140]), np.array([20, 255, 255]))]},
        'Amarillo': {'hsv_range': [(np.array([17, 80, 150]), np.array([40, 255, 255]))]},
        'Verde': {'hsv_range': [(np.array([20, 60, 60]), np.array([75, 255, 255]))]},
        'Azul': {'hsv_range': [(np.array([90, 100, 80]), np.array([130, 255, 255]))]},
        'Violeta': {'hsv_range': [(np.array([110, 60, 80]), np.array([180, 255, 255]))]},
        'Gris': {'hsv_range': [(np.array([0, 0, 50]), np.array([180, 40, 160]))]},
        'Blanco': {'hsv_range': [(np.array([0, 0, 130]), np.array([180, 80, 255]))]},
        'Plata': {'hsv_range': [(np.array([0, 0, 120]), np.array([180, 15, 200]))]},
        'Dorado': {'hsv_range': [(np.array([15, 30, 120]), np.array([35, 180, 255]))]}
    }

def detectar_color_region(region_hsv, config_color):
    """
    Cuenta cuántos píxeles de una región HSV están dentro de un rango de color específico.
    """
    pixeles_totales = 0
    for hsv_bajo, hsv_alto in config_color['hsv_range']:
        mascara = cv2.inRange(region_hsv, hsv_bajo, hsv_alto)
        pixeles_totales += cv2.countNonZero(mascara)
    return pixeles_totales

def identificar_colores_bandas(img_bgr, posiciones_bandas, mascara, definiciones_colores):
    """
    Para cada posición de banda detectada, determina el color dominante según los rangos HSV definidos.
    """ 
    if not posiciones_bandas:
        return []
    #Transforma hsv
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    colores = []
    ancho_banda = 5
    # plt.figure(figsize=(12, 2))
    #Determina region banda para detectar color, horizontal y vertical
    #Lo hace mediante la mascara de la resistencia, si no hay contenido hace un recorte estimativo
    for i, pos_x in enumerate(posiciones_bandas):
        x_inicio = max(0, pos_x - ancho_banda // 2)
        x_fin = min(img_bgr.shape[1], pos_x + ancho_banda // 2)
        coords_y = np.where(mascara[:, pos_x] > 0)[0]
        if len(coords_y) > 0:
            y_inicio, y_fin = coords_y[0], coords_y[-1]
        else:
            y_inicio = img_bgr.shape[0] // 3
            y_fin = img_bgr.shape[0] * 2 // 3
        region_hsv = img_hsv[y_inicio:y_fin, x_inicio:x_fin]
        region_bgr = img_bgr[y_inicio:y_fin, x_inicio:x_fin]
        # Mostrar franja que se analiza para la banda (comentado)
        # plt.subplot(1, len(posiciones_bandas), i+1)
        # plt.imshow(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB))
        # plt.title(f"Banda {i+1}")
        # plt.axis('off')
        mejor_coincidencia = 'Desconocido'
        max_pixeles = 0
        # Recorre los rangos de colores y compara cantidad de pixeles
        for nombre_color, config_color in definiciones_colores.items():
            pixeles = detectar_color_region(region_hsv, config_color)
            if pixeles > max_pixeles:
                max_pixeles = pixeles
                mejor_coincidencia = nombre_color
        if max_pixeles > 0:
            colores.append(mejor_coincidencia)
        else:
            colores.append('Desconocido')
    # plt.suptitle("Franjas analizadas para cada banda")
    # plt.tight_layout()
    # plt.show()
    return colores

def mostrar_valores_hsv_por_banda(img_bgr, posiciones_bandas, mascara):
    """
    Calcula y mouestra el promedio HSV de cada banda, útil para debug/análisis.
    """
    # img_bgr = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    # img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # ancho_banda = 5
    # print("Valores promedio HSV por banda:")
    # for i, pos_x in enumerate(posiciones_bandas):
    #     x_inicio = max(0, pos_x - ancho_banda // 2)
    #     x_fin = min(img_bgr.shape[1], pos_x + ancho_banda // 2)
    #     coords_y = np.where(mascara[:, pos_x] > 0)[0]
    #     if len(coords_y) > 0:
    #         y_inicio, y_fin = coords_y[0], coords_y[-1]
    #     else:
    #         y_inicio = img_bgr.shape[0] // 3
    #         y_fin = img_bgr.shape[0] * 2 // 3
    #     region_hsv = img_hsv[y_inicio:y_fin, x_inicio:x_fin]
    #     h_mean = np.mean(region_hsv[:, :, 0])
    #     s_mean = np.mean(region_hsv[:, :, 1])
    #     v_mean = np.mean(region_hsv[:, :, 2])
    #     print(f"Banda {i+1}: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
    pass

def detectar_bandas_sobel_y_colores(img_cuerpo, plot=True):
    """
    Detecta automáticamente las posiciones de las bandas de colores usando
      el gradiente Sobel (bordes verticales), y llama a la función para identificar colores.
    """
    # --- Detección de picos por Sobel ---
    #Convierte escala de grises y detecta picos con sobel
    gray = cv2.cvtColor(img_cuerpo, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.abs(sobelx)
    sobelx = np.uint8(sobelx)
    proj = np.sum(sobelx, axis=0)
    distance = 3
    prominence = 300

    peaks, _ = find_peaks(proj, distance=distance, prominence=prominence)
    # peaks_originales = list(peaks)  # (Comentado, gráfico innecesario)

    # Deteccion adaptativa
    if len(peaks) < 8:
        #print(f"Se detectaron solo {len(peaks)} picos con prominence={prominence}, reintentando con prominence=150")
        prominence = 150
        peaks, _ = find_peaks(proj, distance=distance, prominence=prominence)
        # peaks_originales = list(peaks)  # (Comentado, gráfico innecesario)

    # Gráficos comentados:
    # if plot:
    #     plt.figure(figsize=(12,4))
    #     plt.subplot(1,2,1)
    #     plt.imshow(cv2.cvtColor(img_cuerpo, cv2.COLOR_BGR2RGB))
    #     for p in peaks_originales:
    #         plt.axvline(x=p, color='r')
    #     plt.title("Picos detectados (bordes verticales, antes de filtrar)")
    #     plt.axis('off')
    #     plt.subplot(1,2,2)
    #     plt.plot(proj)
    #     for p in peaks_originales:
    #         plt.axvline(x=p, color='r', linestyle='--')
    #     plt.title("Proyección vertical del gradiente")
    #     plt.tight_layout()
    #     plt.show()

    peaks = list(peaks)
    if len(peaks) > 8:
        #print(f"Se detectaron {len(peaks)} picos, eliminando los impares a menos de 10 px de su par anterior...")
        i = 1
        while i < len(peaks) - 1:
            if abs(peaks[i+1] - peaks[i]) < 7:
                #print(f"Eliminando pico en posición {i+1} (valor={peaks[i+1]}) por estar muy cerca de {peaks[i]}")
                del peaks[i+1]
            i += 2
        #print("Picos después de eliminar impares demasiado cercanos:", peaks)
    peaks = np.array(peaks)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask // 255
    definiciones_colores = definir_colores_hsv()

    if len(peaks) != 8:
        print("No se detectaron exactamente 8 picos, no se pueden calcular posiciones.")
        return [], []

    # Detecta zona central entre picos y llama a identificar color despues
    posiciones_centrales = []
    for i in range(0, len(peaks)-1, 2):
        x1 = peaks[i]
        x2 = peaks[i+1]
        pos_central = (x1 + x2) // 2
        posiciones_centrales.append(pos_central)

    colores = identificar_colores_bandas(img_cuerpo, posiciones_centrales, mask, definiciones_colores)
    # mostrar_valores_hsv_por_banda(img_cuerpo, posiciones_centrales, mask)  # Comentado, no queremos promedios

    # if plot and posiciones_centrales is not None:
    #     img_show = img_cuerpo.copy()
    #     for x in posiciones_centrales:
    #         cv2.line(img_show, (int(x), 0), (int(x), img_show.shape[0]), (0,255,0), 1)
    #     plt.figure(figsize=(8,4))
    #     plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    #     plt.title("Posiciones centrales de bandas y colores detectados")
    #     plt.axis('off')
    #     plt.show()

    return peaks, colores


def process_all_images():
    """
    Procesa todas las imágenes de INPUT_DIR, aplica rectify_resistor y
    guarda las imágenes en OUTPUT_DIR con sufijo '_out'.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for fname in os.listdir(INPUT_DIR):
        if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png') or fname.lower().endswith('.tif')):
            continue
        in_path = os.path.join(INPUT_DIR, fname)
        warp = rectify_resistor(in_path)
        if warp is None:
            print(f"No se detectó rectángulo azul en {fname}")
            continue
        name, ext = os.path.splitext(fname)
        out_name = f"{name}_out{ext}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), warp)
        print(f"Guardada: {out_name}")

def preprocesar():
    """
    Procesa todas las imágenes ya rectificadas, recorta el cuerpo de la resistencia (preproceso_deteccion) y guarda ese resultado.
    """
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.lower().endswith(('_a_out.png', '_a_out.jpg', '_a_out.tif')):
            continue
        path = os.path.join(OUTPUT_DIR, fname)
        img = cv2.imread(path)
        img_preprocesada = preproceso_deteccion(img)
        if img_preprocesada is None:
            print(f"[AVISO] No se pudo preprocesar {fname}")
            continue
        base_name = os.path.splitext(os.path.basename(fname))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_preprocesada.png")
        cv2.imwrite(out_path, img_preprocesada)
        print(f'\nPreprocesando {base_name}...')
        print(f'Guardada: {base_name}_preprocesada.png')


def calculo_ohms(bandas_colores):
    """ Calcula el valor de una resistencia eléctrica a partir de sus bandas de colores."""
   # if bandas_colores[0] == 'Dorado': 
        #invertir resistencias si el primero es dorado
    #    bandas_colores[0] = bandas_colores[3]
     #   aux = bandas_colores[1]
      #  bandas_colores[1] = bandas_colores[2]
       # bandas_colores[2] = aux

    if len(bandas_colores) < 3:
        return None  # No se puede calcular

    color1, color2, color3 = bandas_colores[:3]
    colores_codigo = {
        "Negro":   (0, 1),
        "Marrón":  (1, 10),
        "Rojo":    (2, 100),
        "Naranja": (3, 1_000),
        "Amarillo":(4, 10_000),
        "Verde":   (5, 100_000),
        "Azul":    (6, 1_000_000),
        "Violeta": (7, 10_000_000),
        "Gris":    (8, 100_000_000),
        "Blanco":  (9, 1_000_000_000)
    }

    try:
        d1 = colores_codigo[color1][0]
        d2 = colores_codigo[color2][0]
        multiplicador = colores_codigo[color3][1]
        valor = (d1 * 10 + d2) * multiplicador
        return valor
    except KeyError:
        return None  # Color inválido
    
def formato_resistencia(valor_ohm):
    """ Formatea un valor de resistencia en ohmios a una representación con unidades legibles."""
    if valor_ohm >= 1_000_000:
        return f"{valor_ohm / 1_000_000:.1f} MΩ"
    elif valor_ohm >= 1_000:
        return f"{valor_ohm / 1_000:.1f} kΩ"
    else:
        return f"{valor_ohm} Ω"

def main():
    """
    Rectifica todas las imágenes originales.
    Recorta el cuerpo en todas las imágenes rectificadas.
    Detecta y muestra los colores de las bandas solo en las imágenes del tipo "a" ya preprocesadas.
    """
    process_all_images()        # Rectificar imágenes
    preprocesar()         # Recortar cuerpo

    # Detección de bandas sobre las preprocesadas:
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.lower().endswith(('_preprocesada.png', '_preprocesada.jpg', '_preprocesada.tif')):
            continue
        if "a" not in fname:
            continue
        path = os.path.join(OUTPUT_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"No se pudo cargar la imagen: {path}")
            continue
        #Detecta bandas
        picos_colores = detectar_bandas_sobel_y_colores(img, plot=True)
        if (
            picos_colores is None
            or picos_colores[0] is None
            or picos_colores[1] is None
        ):
            print(f"Saltando imagen {fname} porque no se detectaron 8 picos.")
            continue
        
        picos, bandas_colores = picos_colores
        if bandas_colores[-1] != 'Dorado': #invierto los colores si el ultimo no es dorado
            bandas_colores = list(reversed(bandas_colores))

        print(f"\nImagen {fname} - Bandas detectadas:", bandas_colores)
        ohms = calculo_ohms(bandas_colores)
        resultado = formato_resistencia(ohms)
        print(f"La resistencia {fname} es de {resultado}")

if __name__ == "__main__":
    main()
