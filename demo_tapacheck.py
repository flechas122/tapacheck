# TapaCheck - demo de inspeccion en vivo
#
# 1. Carga el pipeline que entreno el notebook (modelos/tapacheck.joblib)
# 2. Toma fotos de botellas (por defecto valid + test del dataset, que el modelo nunca vio)
# 3. Recorta siempre el mismo cuadro (camara fija), muestra el veredicto ROSCADA / MAL ROSCADA y el tiempo que tardo
#
# Teclas: ESPACIO = siguiente botella, ESC (o la X de la ventana) = salir
# Uso:    python demo_tapacheck.py                 (fotos de valid + test del dataset)
#         python demo_tapacheck.py foto.jpg        (una foto)
#         python demo_tapacheck.py carpeta/        (todas las fotos de una carpeta)
# Las fotos propias deben estar encuadradas como las del dataset: la tapa en el centro, vista de lado.

import os
import sys
import glob
import time
import random
import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

RUTA_MODELO = os.path.join("modelos", "tapacheck.joblib")
RUTA_DATOS = "data"
extensiones = ('.jpg', '.jpeg', '.png', '.bmp')
VENTANA = "TapaCheck - inspeccion (ESPACIO: siguiente, ESC: salir)"

# 1. Cargar el pipeline entrenado
if not os.path.exists(RUTA_MODELO):
    print("No encuentro", RUTA_MODELO, "- ejecute primero el notebook completo para entrenar y guardar el modelo.")
    sys.exit(1)
print("Cargando modelo...")
pipeline = joblib.load(RUTA_MODELO)
modelo = pipeline['modelo']
descriptor = pipeline['descriptor']
TAMANO = pipeline['tamano']
ROI = pipeline['roi']
CLASES = pipeline['clases']
MAPA_CLASES = pipeline['mapa_clases']
NOMBRES = pipeline['nombres']
print("Modelo listo:", descriptor, "+", type(modelo).__name__)

# 2. Mismo recorte y mismos descriptores del notebook (deben ser identicos a los del entrenamiento)
def cuadro_roi(forma):
    alto, ancho = forma[:2]
    lado = int(round(ROI['lado'] * ancho))
    x0, y0 = int(round(ROI['cx'] * ancho - lado / 2)), int(round(ROI['cy'] * alto - lado / 2))
    return x0, y0, x0 + lado, y0 + lado

def recortar_tapa(gris):
    x0, y0, x1, y1 = cuadro_roi(gris.shape)
    borde = max(0, -x0, -y0, x1 - gris.shape[1], y1 - gris.shape[0])
    if borde:
        gris = cv2.copyMakeBorder(gris, borde, borde, borde, borde, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + borde, y0 + borde, x1 + borde, y1 + borde
    return cv2.resize(gris[y0:y1, x0:x1], TAMANO, interpolation=cv2.INTER_AREA)

def extraer_hog(img):
    return hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

def extraer_lbp(img, P=8, R=1):
    lbp = local_binary_pattern(img, P=P, R=R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float32")
    return hist / (hist.sum() + 1e-7)

def extraer_glcm(img):
    glcm = graycomatrix((img // 4).astype(np.uint8), distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64, symmetric=True, normed=True)
    vector = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        vector.extend(graycoprops(glcm, prop).ravel())
    return np.array(vector)

def inspeccionar(gris):
    t0 = time.time()
    img = recortar_tapa(gris)
    if descriptor == 'HOG':
        X = pipeline['escaladores']['HOG'].transform([extraer_hog(img)])
    elif descriptor == 'LBP':
        X = pipeline['escaladores']['LBP'].transform([extraer_lbp(img)])
    elif descriptor == 'GLCM':
        X = pipeline['escaladores']['GLCM'].transform([extraer_glcm(img)])
    else:
        F = np.hstack([extraer_hog(img), extraer_lbp(img), extraer_glcm(img)])
        X = pipeline['scaler_fusion'].transform([F])
        if descriptor == 'Fusión + PCA':
            X = pipeline['pca'].transform(X)
    pred = CLASES[int(modelo.predict(X)[0])]
    return pred, 1000 * (time.time() - t0)

# 3. Etiqueta real, si la foto trae la de Roboflow: <split>/images/foto.jpg -> <split>/labels/foto.txt
#    Devuelve ROSCADA / MAL ROSCADA, 'fuera' si es una clase que TapaCheck no revisa, o None si no hay etiqueta.
def etiqueta_real(ruta):
    ruta_txt = os.path.join(os.path.dirname(os.path.dirname(ruta)), 'labels',
                            os.path.splitext(os.path.basename(ruta))[0] + '.txt')
    if not os.path.exists(ruta_txt):
        return None
    with open(ruta_txt) as f:
        lineas = [l for l in f.read().splitlines() if l.strip()]
    if len(lineas) != 1:
        return None
    nombre = NOMBRES[int(lineas[0].split()[0])]
    return CLASES[MAPA_CLASES[nombre]] if nombre in MAPA_CLASES else 'fuera'

# 4. Lista de fotos a inspeccionar
if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    fotos = [sys.argv[1]]
elif len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
    fotos = [os.path.join(sys.argv[1], f) for f in sorted(os.listdir(sys.argv[1])) if f.lower().endswith(extensiones)]
else:
    fotos = []
    for actual, _, archivos in os.walk(RUTA_DATOS):
        if 'data.yaml' in archivos:
            for split in ['valid', 'test']:
                fotos += [r for r in glob.glob(os.path.join(actual, split, 'images', '*')) if r.lower().endswith(extensiones)]
    fotos = [r for r in fotos if etiqueta_real(r) in CLASES]   # solo tapas que TapaCheck revisa
    random.shuffle(fotos)
if not fotos:
    print("No hay fotos para inspeccionar. Descomprima el dataset en data/ o pase una foto o carpeta.")
    sys.exit(1)

# 5. Ciclo de inspeccion
def ventana_abierta():
    try:
        return cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False

def esperar_tecla():
    # Espera una tecla. Si se cierra la ventana con la X cuenta como ESC; si no, el siguiente imshow la vuelve a abrir.
    while True:
        tecla = cv2.waitKey(50)
        if tecla != -1:
            return tecla & 0xFF
        if not ventana_abierta():
            return 27

aciertos, total, calentado = 0, 0, False
for ruta in fotos:
    foto = cv2.imread(ruta)
    if foto is None:
        continue
    gris = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
    if not calentado:  # la primera llamada siempre es mas lenta
        inspeccionar(gris)
        calentado = True

    pred, ms = inspeccionar(gris)
    real = etiqueta_real(ruta)
    color = (60, 160, 60) if pred == 'ROSCADA' else (40, 60, 220)   # BGR: verde / rojo
    escala = 600 / max(foto.shape[:2])
    frame = cv2.resize(foto, None, fx=escala, fy=escala)
    x0, y0, x1, y1 = cuadro_roi(frame.shape)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 4)
    frame = cv2.copyMakeBorder(frame, 0, 600 - frame.shape[0], 0, 600 - frame.shape[1], cv2.BORDER_CONSTANT, value=(39, 33, 28))
    cv2.rectangle(frame, (0, 0), (599, 599), color, 12)
    panel = np.full((150, 600, 3), (39, 33, 28), np.uint8)
    cv2.putText(panel, pred, (20, 70), cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 3)
    cv2.putText(panel, f"{ms:.0f} ms", (460, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
    if real in CLASES:
        total += 1
        aciertos += (pred == real)
        cv2.putText(panel, f"real: {real}   aciertos: {aciertos}/{total}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
    else:
        cv2.putText(panel, os.path.basename(ruta)[:40], (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    print(f"{os.path.basename(ruta)[:40]:40s} -> {pred:11s} ({ms:.0f} ms)" + (f"  real: {real}" if real in CLASES else ""))

    cv2.imshow(VENTANA, np.vstack([panel, frame]))
    if esperar_tecla() == 27:   # ESC o la X de la ventana
        break

cv2.destroyAllWindows()
if total:
    print(f"\nAciertos: {aciertos} de {total} tapas ({100 * aciertos / total:.1f} %)")
