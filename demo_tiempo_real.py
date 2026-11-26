# TapaCheck 2.0 - comprobacion en tiempo real (Entrega 4)
#
# Red preentrenada congelada (embeddings) + AutoGluon, con lo que entreno TapaCheck_E4_transfer_learning.ipynb:
#   modelos/tapacheck_e4.json (configuracion) y modelos/autogluon_e4_despliegue/ (cabeza de decision)
#
# Uso:  python demo_tiempo_real.py                 flujo continuo con las fotos de valid + test (simula la banda)
#       python demo_tiempo_real.py --camara 0      camara del computador, en vivo
#       python demo_tiempo_real.py carpeta/        flujo con las fotos de una carpeta (o una sola foto)
# Opciones: --intervalo 700  milisegundos que se ve cada foto del flujo (0 = lo mas rapido posible)
#           --limite 50      parar despues de N fotos o cuadros
# Teclas:   ESPACIO = pausa / continuar, ESC (o la X de la ventana) = salir
# Con la camara: la tapa va dentro del recuadro, vista de lado y encuadrada como en las fotos del dataset.

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')   # oculta los mensajes informativos de TensorFlow
import sys
import glob
import json
import time
import random
import argparse
from collections import deque

import cv2
import yaml
import numpy as np
import pandas as pd

RUTA_CONFIG = os.path.join("modelos", "tapacheck_e4.json")
RUTA_DATOS = "data"
extensiones = ('.jpg', '.jpeg', '.png', '.bmp')
VENTANA = "TapaCheck 2.0 - tiempo real (ESPACIO: pausa, ESC: salir)"
BOTELLAS_HORA = 2_400   # velocidad de la linea de ejemplo (la misma del notebook)

parser = argparse.ArgumentParser(description="TapaCheck 2.0: inspeccion de tapas en tiempo real")
parser.add_argument('fuente', nargs='?', help="foto o carpeta de fotos (por defecto valid + test del dataset)")
parser.add_argument('--camara', type=int, default=None, help="indice de la camara (0 = la del computador)")
parser.add_argument('--intervalo', type=int, default=700, help="ms que se ve cada foto del flujo (0 = sin espera)")
parser.add_argument('--limite', type=int, default=0, help="parar despues de N fotos o cuadros (0 = sin limite)")
args = parser.parse_args()

# 1. Configuracion y modelos
if not os.path.exists(RUTA_CONFIG):
    print("No encuentro", RUTA_CONFIG, "- ejecute primero TapaCheck_E4_transfer_learning.ipynb completo.")
    sys.exit(1)
with open(RUTA_CONFIG, encoding='utf-8') as f:
    config = json.load(f)
ROI, ESTADOS, CLASES = config['roi'], config['estados'], config['clases']
TAMANO = tuple(config['tamano'])
COLUMNAS = [f'f_{i}' for i in range(config['columnas'])]
MAPA_ESTADOS = config.get('mapa_estados', {'good_cap': 'Roscada', 'open_cap': 'Floja',
                                           'misplaced_cap': 'Ladeada', 'no_cap': 'Sin tapa'})

print("Cargando TensorFlow y AutoGluon...")
import tensorflow as tf
from tensorflow import keras
from autogluon.tabular import TabularPredictor

REDES = {
    'EfficientNetB0': (keras.applications.EfficientNetB0, keras.applications.efficientnet.preprocess_input),
    'MobileNetV2': (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
    'ResNet50': (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
    'DenseNet121': (keras.applications.DenseNet121, keras.applications.densenet.preprocess_input),
}
constructor, preprocesar = REDES[config['red']]
extractor = constructor(include_top=False, weights='imagenet', input_shape=TAMANO + (3,), pooling='avg')
extractor.trainable = False   # congelada, igual que en el notebook
predictor = TabularPredictor.load(config['autogluon'], verbosity=0)
predictor.persist()           # modelos en memoria para responder rapido
NOMBRE_SISTEMA = f"{config['red']} + AutoGluon"
print(f"Modelo listo: {config['red']} congelada + AutoGluon ({predictor.model_best})")


# 2. Mismo recorte fijo y mismo preprocesamiento del notebook.
#    El cuadro esta en fracciones de la foto: en las fotos del dataset (640x640) es el mismo cuadrado del notebook;
#    en un cuadro de camara de otra proporcion equivale a estirarlo a cuadrado, como hizo Roboflow con las fotos.
@tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.uint8)])
def embedding(img):
    alto, ancho = tf.shape(img)[0], tf.shape(img)[1]
    alto_f, ancho_f = tf.cast(alto, tf.float32), tf.cast(ancho, tf.float32)
    lado_x = tf.cast(tf.round(ROI['lado'] * ancho_f), tf.int32)
    lado_y = tf.cast(tf.round(ROI['lado'] * alto_f), tf.int32)
    x0 = tf.cast(tf.round(ROI['cx'] * ancho_f - tf.cast(lado_x, tf.float32) / 2), tf.int32)
    y0 = tf.cast(tf.round(ROI['cy'] * alto_f - tf.cast(lado_y, tf.float32) / 2), tf.int32)
    borde = tf.maximum(0, tf.reduce_max(tf.stack([-x0, -y0, x0 + lado_x - ancho, y0 + lado_y - alto])))
    img = tf.pad(img, [[borde, borde], [borde, borde], [0, 0]], mode='SYMMETRIC')
    recorte = tf.image.crop_to_bounding_box(img, y0 + borde, x0 + borde, lado_y, lado_x)
    x = preprocesar(tf.image.resize(recorte, TAMANO, method='area'))
    return extractor(x[tf.newaxis], training=False)[0]


def inspeccionar(img_rgb):
    # Cuadro RGB -> recorte fijo -> red congelada -> AutoGluon -> estado, veredicto, probabilidades y milisegundos
    t0 = time.perf_counter()
    fila = pd.DataFrame([embedding(img_rgb).numpy()], columns=COLUMNAS)
    proba = predictor.predict_proba(fila).iloc[0][ESTADOS].to_numpy(dtype=float)
    estado = ESTADOS[int(np.argmax(proba))]
    return estado, CLASES[int(estado != 'Roscada')], proba, 1000 * (time.perf_counter() - t0)


def cuadro_roi(alto, ancho):
    lado_x, lado_y = int(round(ROI['lado'] * ancho)), int(round(ROI['lado'] * alto))
    x0, y0 = int(round(ROI['cx'] * ancho - lado_x / 2)), int(round(ROI['cy'] * alto - lado_y / 2))
    return x0, y0, x0 + lado_x, y0 + lado_y


# 3. Estado real, si la foto trae la etiqueta de Roboflow: <split>/images/foto.jpg -> <split>/labels/foto.txt
def leer_nombres():
    if 'nombres' in config:
        return config['nombres']
    for actual, _, archivos in os.walk(RUTA_DATOS):
        if 'data.yaml' in archivos:
            with open(os.path.join(actual, 'data.yaml'), encoding='utf-8') as f:
                nombres = yaml.safe_load(f)['names']
            return list(nombres.values()) if isinstance(nombres, dict) else list(nombres)
    return None

NOMBRES = leer_nombres()

def estado_real(ruta):
    # Devuelve el estado, 'fuera' si es una clase que TapaCheck no revisa, o None si no hay etiqueta
    if NOMBRES is None:
        return None
    ruta_txt = os.path.join(os.path.dirname(os.path.dirname(ruta)), 'labels',
                            os.path.splitext(os.path.basename(ruta))[0] + '.txt')
    if not os.path.exists(ruta_txt):
        return None
    with open(ruta_txt) as f:
        lineas = [l for l in f.read().splitlines() if l.strip()]
    if len(lineas) != 1:
        return None
    return MAPA_ESTADOS.get(NOMBRES[int(lineas[0].split()[0])], 'fuera')


# 4. Pantalla: foto con el recuadro a la izquierda, veredicto y probabilidades a la derecha
FONDO = (39, 33, 28)
VERDE, ROJO, AMBAR = (60, 160, 60), (40, 60, 220), (0, 190, 255)   # BGR
GRIS, BLANCO, TENUE = (120, 120, 120), (235, 235, 235), (170, 170, 170)
ANCHO_PANTALLA, ALTO_PANTALLA, LADO_FOTO = 1060, 640, 600


def texto(img, cadena, x, y, escala=0.7, color=BLANCO, grosor=1, fuente=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, cadena, (x, y), fuente, escala, color, grosor, cv2.LINE_AA)


def dibujar(frame_bgr, estado, veredicto, proba, ms, linea_tiempo, contadores, real=None):
    pantalla = np.full((ALTO_PANTALLA, ANCHO_PANTALLA, 3), FONDO, np.uint8)
    color = VERDE if veredicto == 'ROSCADA' else ROJO

    # Foto (o cuadro de camara) con el recuadro que mira el sistema
    alto, ancho = frame_bgr.shape[:2]
    escala = LADO_FOTO / max(alto, ancho)
    foto = cv2.resize(frame_bgr, (int(round(ancho * escala)), int(round(alto * escala))), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = cuadro_roi(*foto.shape[:2])
    cv2.rectangle(foto, (x0, y0), (x1, y1), color, 3)
    oy, ox = 20 + (LADO_FOTO - foto.shape[0]) // 2, 20 + (LADO_FOTO - foto.shape[1]) // 2
    pantalla[oy:oy + foto.shape[0], ox:ox + foto.shape[1]] = foto
    cv2.rectangle(pantalla, (ox - 6, oy - 6), (ox + foto.shape[1] + 5, oy + foto.shape[0] + 5), color, 6)

    # Panel derecho
    px = LADO_FOTO + 50
    texto(pantalla, "TapaCheck 2.0", px, 50, 0.9, BLANCO, 2)
    texto(pantalla, NOMBRE_SISTEMA, px, 80, 0.55, TENUE)
    texto(pantalla, veredicto, px, 150, 1.45, color, 3, cv2.FONT_HERSHEY_DUPLEX)
    i = ESTADOS.index(estado)
    texto(pantalla, f"Estado: {estado} ({100 * proba[i]:.0f} %)", px, 190, 0.75, BLANCO, 2)
    if real in ESTADOS:
        if real == estado:
            nota, tono = "acierto", VERDE
        elif CLASES[int(real != 'Roscada')] == veredicto:
            nota, tono = "otro estado, veredicto correcto", AMBAR
        else:
            nota, tono = "error", ROJO
        texto(pantalla, f"Real: {real} ({nota})", px, 220, 0.6, tono, 1)

    # Probabilidad de cada estado
    texto(pantalla, "Probabilidad por estado", px, 265, 0.55, TENUE)
    for j, (nombre, p) in enumerate(zip(ESTADOS, proba)):
        y = 295 + 38 * j
        texto(pantalla, nombre, px, y + 16, 0.6, BLANCO)
        cv2.rectangle(pantalla, (px + 105, y), (px + 315, y + 20), (70, 62, 55), -1)
        largo = int(round(210 * float(p)))
        if largo:
            barra = (VERDE if nombre == 'Roscada' else ROJO) if j == i else GRIS
            cv2.rectangle(pantalla, (px + 105, y), (px + 105 + largo, y + 20), barra, -1)
        texto(pantalla, f"{100 * p:3.0f} %", px + 325, y + 16, 0.55, BLANCO)

    # Tiempo y contadores
    texto(pantalla, f"Latencia: {ms:.0f} ms por botella", px, 455, 0.65, BLANCO, 2)
    texto(pantalla, linea_tiempo, px, 485, 0.55, TENUE)
    texto(pantalla, f"Inspeccionadas: {contadores['total']}   Apartadas: {contadores['apartadas']}", px, 530, 0.6, BLANCO)
    n = contadores['con_etiqueta']
    if n:
        texto(pantalla, f"Veredicto correcto: {contadores['veredictos_ok']} de {n} ({porcentaje(contadores['veredictos_ok'], n)})",
              px, 562, 0.6, BLANCO)
        texto(pantalla, f"Estado correcto: {contadores['aciertos']} de {n} ({porcentaje(contadores['aciertos'], n)})",
              px, 592, 0.6, TENUE)
    texto(pantalla, "ESPACIO: pausa   ESC: salir", px, 628, 0.5, TENUE)
    return pantalla


def porcentaje(parte, total):
    return f"{100 * parte / total:.1f} %".replace('.', ',')


def con_pausa(pantalla):
    copia = pantalla.copy()
    texto(copia, "PAUSA", 40, 70, 1.4, (0, 215, 255), 3, cv2.FONT_HERSHEY_DUPLEX)
    return copia


# 5. Ventana: la X cuenta como ESC (si no se revisa, el siguiente imshow la vuelve a abrir)
def ventana_abierta():
    try:
        return cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def leer_tecla(espera_ms):
    # Devuelve 'salir', 'pausa' o None
    tecla = cv2.waitKey(max(1, espera_ms))
    if tecla != -1:
        tecla &= 0xFF
        if tecla == 27:
            return 'salir'
        if tecla == 32:
            return 'pausa'
    return None if ventana_abierta() else 'salir'


# 6. Fuente de imagenes
def lista_de_fotos():
    if args.fuente and os.path.isfile(args.fuente):
        return [args.fuente]
    if args.fuente and os.path.isdir(args.fuente):
        return [os.path.join(args.fuente, f) for f in sorted(os.listdir(args.fuente)) if f.lower().endswith(extensiones)]
    fotos = []
    for actual, _, archivos in os.walk(RUTA_DATOS):
        if 'data.yaml' in archivos:
            for split in ['valid', 'test']:
                fotos += [r for r in glob.glob(os.path.join(actual, split, 'images', '*')) if r.lower().endswith(extensiones)]
    fotos = [r for r in sorted(fotos) if estado_real(r) in ESTADOS]   # solo tapas que TapaCheck revisa
    random.shuffle(fotos)
    return fotos


contadores = {'total': 0, 'apartadas': 0, 'aciertos': 0, 'veredictos_ok': 0, 'con_etiqueta': 0}
latencias = []


def registrar(estado, veredicto, ms, real):
    contadores['total'] += 1
    contadores['apartadas'] += veredicto != 'ROSCADA'
    if real in ESTADOS:
        contadores['con_etiqueta'] += 1
        contadores['aciertos'] += estado == real
        contadores['veredictos_ok'] += veredicto == CLASES[int(real != 'Roscada')]
    latencias.append(ms)


# Calentamiento: la primera llamada prepara el grafo de TensorFlow y los modelos de AutoGluon
inspeccionar(tf.zeros((640, 640, 3), tf.uint8))
inspeccionar(tf.zeros((480, 640, 3), tf.uint8))
cv2.namedWindow(VENTANA, cv2.WINDOW_AUTOSIZE)

if args.camara is not None:
    # --- Camara en vivo: cada cuadro se inspecciona en cuanto llega ---
    cap = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara {args.camara}. Pruebe con otro indice (--camara 1) o use el flujo de fotos.")
        sys.exit(1)
    print("Camara abierta. Ponga la tapa dentro del recuadro, vista de lado.")
    ciclos, pausa, pantalla = deque(maxlen=30), False, None
    t_anterior = time.perf_counter()
    while True:
        if not pausa:
            ok, frame = cap.read()
            if not ok:
                print("La camara dejo de enviar imagen.")
                break
            estado, veredicto, proba, ms = inspeccionar(tf.convert_to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            registrar(estado, veredicto, ms, None)
            ahora = time.perf_counter()
            ciclos.append(ahora - t_anterior)
            t_anterior = ahora
            fps = len(ciclos) / sum(ciclos)
            pantalla = dibujar(frame, estado, veredicto, proba, ms, f"{fps:.1f} cuadros por segundo en vivo", contadores)
            cv2.imshow(VENTANA, pantalla)
        accion = leer_tecla(1)
        if accion == 'salir' or (args.limite and contadores['total'] >= args.limite):
            break
        if accion == 'pausa':
            pausa = not pausa
            cv2.imshow(VENTANA, con_pausa(pantalla) if pausa else pantalla)
            t_anterior = time.perf_counter()
    cap.release()
else:
    # --- Flujo continuo de fotos: simula la banda transportadora ---
    fotos = lista_de_fotos()
    if not fotos:
        print("No hay fotos para inspeccionar. Descomprima el dataset en data/ o pase una foto o carpeta.")
        sys.exit(1)
    if args.limite:
        fotos = fotos[:args.limite]
    print(f"Flujo de {len(fotos)} fotos, una cada {args.intervalo} ms (ESPACIO: pausa, ESC: salir)")
    salir = False
    for ruta in fotos:
        img = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
        estado, veredicto, proba, ms = inspeccionar(img)
        real = estado_real(ruta)
        registrar(estado, veredicto, ms, real)
        capacidad = 1000 / np.median(latencias[-20:])
        linea = f"Capacidad: {capacidad:.1f} botellas/s (la linea pide {BOTELLAS_HORA / 3600:.2f})".replace('.', ',')
        pantalla = dibujar(cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR), estado, veredicto, proba, ms, linea, contadores, real)
        print(f"{os.path.basename(ruta)[:40]:40s} -> {veredicto:11s} {estado:8s} {100 * proba.max():3.0f} %  ({ms:.0f} ms)"
              + (f"  real: {real}" if real in ESTADOS else ""))
        cv2.imshow(VENTANA, pantalla)

        # Espera el intervalo del flujo; en pausa espera hasta que se pulse ESPACIO otra vez
        fin, pausa = time.perf_counter() + args.intervalo / 1000, False
        while True:
            accion = leer_tecla(15 if args.intervalo else 1)
            if accion == 'salir':
                salir = True
                break
            if accion == 'pausa':
                pausa = not pausa
                cv2.imshow(VENTANA, con_pausa(pantalla) if pausa else pantalla)
                fin = time.perf_counter() + args.intervalo / 1000
            if not pausa and time.perf_counter() >= fin:
                break
        if salir:
            break

cv2.destroyAllWindows()
if latencias:
    print(f"\nInspeccionadas: {contadores['total']} | apartadas: {contadores['apartadas']} | "
          f"latencia mediana {np.median(latencias):.0f} ms, percentil 95 {np.percentile(latencias, 95):.0f} ms")
if contadores['con_etiqueta']:
    n = contadores['con_etiqueta']
    print(f"Veredicto correcto: {contadores['veredictos_ok']} de {n} ({porcentaje(contadores['veredictos_ok'], n)}) | "
          f"estado correcto: {contadores['aciertos']} de {n} ({porcentaje(contadores['aciertos'], n)})")
