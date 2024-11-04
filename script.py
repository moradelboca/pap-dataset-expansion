import torch
from PIL import Image
import numpy as np
import os
import pandas as pd
from progress_bar import ProgressBar
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def cortar_imagen_en_ejes(imagen, divisiones_x, divisiones_y):
    ancho, alto = imagen.size
    ancho_div = ancho // divisiones_x
    alto_div = alto // divisiones_y
    imagenes = []
    limites = []
    for x in range(divisiones_x):
        for y in range(divisiones_y):
            xmin = x * ancho_div
            ymin = y * alto_div
            xmax = (x + 1) * ancho_div
            ymax = (y + 1) * alto_div
            imagen_recortada = imagen.crop(
                (xmin, ymin, xmax, ymax))
            limites.append({"xmin": xmin, "ymin": ymin,
                           "xmax": xmax, "ymax": ymax})
            imagenes.append(imagen_recortada)
    return imagenes, limites

def generar_heatmap(imagen, centros, divisiones=50, mostrar=False):
    ancho, alto = imagen.size
    # Harcodeado para testing
    heatmap = np.zeros((divisiones, divisiones))
    for centro in centros:
        x, y = centro
        x = int(x / ancho * divisiones)
        y = int(y / alto * divisiones)
        heatmap[y, x] += 1
    if mostrar:
        directorio = './dataset_expandido/heatmaps'
        if not os.path.exists(directorio):
            os.mkdir(directorio)

        # Overlay the dots
        for centro in centros:
            x, y = centro
            x = x / ancho * divisiones
            y = y / alto * divisiones
            plt.scatter(x - 0.5, y - 0.5, color='blue')

        # Mostrar el heatmap
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        nombre = imagen.filename.split('\\')[-1].split('.')[0]
        plt.savefig(f'{directorio}/heatmap{nombre}.png')
    return heatmap


def verificar_limites(imagen, tamanio_salida, recorte):
    ancho_entrada, alto_entrada = imagen.size
    ancho_salida, alto_salida = tamanio_salida
    xmin, ymin, xmax, ymax = recorte
    if xmin < 0:
        xmin = 0
        xmax = ancho_salida
    if ymin < 0:
        ymin = 0
        ymax = alto_salida
    if xmax > ancho_entrada:
        xmax = ancho_entrada
        xmin = ancho_entrada - ancho_salida
    if ymax > alto_entrada:
        ymax = alto_entrada
        ymin = alto_entrada - alto_salida
    return xmin, ymin, xmax, ymax


def obtener_iou(coordenadasImg1, coordenadasImg2):
    # Calcular el area de la interseccion
    xA = max(coordenadasImg1[0], coordenadasImg2[0])
    yA = max(coordenadasImg1[1], coordenadasImg2[1])
    xB = min(coordenadasImg1[2], coordenadasImg2[2])
    yB = min(coordenadasImg1[3], coordenadasImg2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # Calcular el area de las imagenes
    areaImg1 = (coordenadasImg1[2] - coordenadasImg1[0] + 1) * \
        (coordenadasImg1[3] - coordenadasImg1[1] + 1)
    areaImg2 = (coordenadasImg2[2] - coordenadasImg2[0] + 1) * \
        (coordenadasImg2[3] - coordenadasImg2[1] + 1)
    # Calcular el area de la union
    unionArea = areaImg1 + areaImg2 - interArea
    # Calcular la interseccion sobre la union
    iou = interArea / unionArea
    return iou


def expansion_dataset_divisiones_uniformes(umbral_celulas=0):
    # Cargar las imágenes del dataset
    img_dir = './dataset'
    img_list = os.listdir(img_dir)

    if not os.path.exists('./dataset_expandido'):
        os.mkdir('./dataset_expandido')

    if umbral_celulas > 0:
        modelo = cargar_modelo()

    bar = ProgressBar(len(img_list)*6)

    # Abrir el csv con las anotaciones
    clases = pd.read_csv('./dataset/_classes.csv')
    # Dataframe donde voy a ir guardando todas las nuevas clases de cada imagen
    nuevas_clases = pd.DataFrame(columns=['filename', 'Sospechoso'])

    # Procesar todas las imagenes
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)

        # Para que no se frene la ejecucion en caso de que
        # algun archivo no sea imagen
        try:
            img = Image.open(img_path)
        except:
            continue

        # Si el umbral de celulas es mayor a 0, entonces
        # se debe detectar celulas en la imagen
        if umbral_celulas:
            # Convertir la imagen a un array de numpy
            img_array = np.array(img)
            # Detectar celulas en la imagen
            resultado = modelo(img_array)
            # Pasarlo a pandas para poder acceder a los valores
            df_resultado = resultado.pandas().xyxy[0]

        # Obtener clase de la imagen.
        # IMPORTANTE: Al parecer en el dataset sospechoso tiene un espacio adelante.
        # Pasa lo mismo con normal.
        clase = clases[clases['filename'] == img_name][' Sospechoso'].values[0]

        # Hacemos 6 divisiones de la imagen
        imagenes, limites = cortar_imagen_en_ejes(img, 3, 2)
        for imagen_index, imagen in enumerate(imagenes):
            bar.display()

            if (umbral_celulas):
                # Contar la cantidad de celulas dentro del recorte
                # teniendo en cuenta los bounds del recorte
                # y el resultado de la detección de celulas
                celulas = 0
                for celula_index in range(len(df_resultado)):
                    if (df_resultado['xmin'][celula_index] >= limites[imagen_index]['xmin'] and
                            df_resultado['ymin'][celula_index] >= limites[imagen_index]['ymin'] and
                            df_resultado['xmax'][celula_index] <= limites[imagen_index]['xmax'] and
                            df_resultado['ymax'][celula_index] <= limites[imagen_index]['ymax']):
                        celulas += 1
                        # Para no seguir iterando si ya se superó el umbral
                        if celulas >= umbral_celulas:
                            break
                # Si no se supera el umbral de celulas, no se guarda la imagen
                if celulas < umbral_celulas:
                    continue

            # Guardar la imagen
            nuevo_nombre = f'{img_name}_{imagen_index}.jpg'
            imagen.save(f'./dataset_expandido/{nuevo_nombre}')
            # Guardar la clase de la imagen
            nuevas_clases = nuevas_clases._append(
                {'filename': nuevo_nombre, 'Sospechoso': clase}, ignore_index=True)

    # Guardar las nuevas clases en un csv
    nuevas_clases.to_csv('./dataset_expandido/_classes.csv', index=False)


def expansion_dataset_heatmap(ancho_salida, alto_salida, offset=0.5, iou_max=0.5, umbral_celulas=0):
    img_dir = './dataset'
    img_list = os.listdir(img_dir)

    if not os.path.exists('./dataset_expandido'):
        os.mkdir('./dataset_expandido')

    modelo = cargar_modelo()

    bar = ProgressBar(len(img_list) - 1) # -1 porque hay un archivo que no es imagen

    clases = pd.read_csv(img_dir + '/_classes.csv')
    nuevas_clases = pd.DataFrame(columns=['filename', 'Sospechoso'])

    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)

        try:
            img = Image.open(img_path) 
            clase = clases[clases['filename'] == img_name][' Sospechoso'].values[0]
        except:
            continue
        
        ancho_entrada, alto_entrada = img.size
        img_array = np.array(img)
        resultado = modelo(img_array)
        df_resultado = resultado.pandas().xyxy[0]
        # Mostrar las detecciones
        #resultado.show()
        
        # Obtener los centros de los bounding boxes
        centros = [ (int((df_resultado["xmin"][i] + df_resultado["xmax"][i]) / 2), 
                    int((df_resultado["ymin"][i] + df_resultado["ymax"][i]) / 2))
                   for i in range(len(df_resultado))]
        divisiones = 5
        heatmap = generar_heatmap(img, centros, divisiones, mostrar=False)
        if (umbral_celulas):
            if np.max(heatmap) < umbral_celulas:
                continue
            else:
                # Encontrar los indices de los puntos que superen el umbral
                indices = np.where(heatmap >= umbral_celulas)
        else:
            # Encontrar los indices mas altos
            indices = np.where(heatmap == np.max(heatmap))
        # Guarda las imagenes que se extraen de una mas grande. Sirve para calcular que no
        # se solapen
        imagenes_guardadas = []

        # Realizar extraccion en todos las regiones que cumplan con el umbral
        for i in range(len(indices[0])):
            # Centro de la region
            region_x = indices[1][i]
            region_y = indices[0][i]
            x = (region_x + 0.5) * ancho_entrada / divisiones
            y = (region_y + 0.5) * alto_entrada / divisiones
            # Validar que la imagen inicial entre en los limites centrada en x, y
            if (x - (ancho_salida // 2) < 0):
                x = ancho_salida // 2
            if (y - alto_salida // 2 < 0):
                y = alto_salida // 2
            if (x + ancho_salida // 2 > ancho_entrada):
                x = ancho_entrada - ancho_salida // 2
            if (y + alto_salida // 2 > alto_entrada):
                y = alto_entrada - alto_salida // 2
            # Generar imagenes con el offset en todas las direcciones
            for indice_x in [0, 1, -1]:
                for indice_y in [0, 1, -1]:
                    # Evitar diagonales (por ahora)
                    if indice_x == 1 and indice_y == 1 \
                    or indice_x == -1 and indice_y == -1 \
                    or indice_x == 1 and indice_y == -1 \
                    or indice_x == -1 and indice_y == 1:
                        continue
                    # Calcular los limites de la imagen
                    xmin = x - ancho_salida // 2 + ancho_salida * offset * indice_x
                    ymin = y - alto_salida // 2 + alto_salida * offset * indice_y
                    xmax = x + ancho_salida // 2 + ancho_salida * offset * indice_x
                    ymax = y + alto_salida // 2 + alto_salida * offset * indice_y
                    # Verificar que los limites no se salgan de la imagen
                    xmin, ymin, xmax, ymax = verificar_limites(img, (ancho_salida, alto_salida),(xmin, ymin, xmax, ymax))
                    # Validar que la imagen no se solape teniendo en cuenta el IoU_max
                    solapada = False
                    for imagen_guardada in imagenes_guardadas:
                        if obtener_iou((xmin, ymin, xmax, ymax), imagen_guardada) > iou_max:
                            solapada = True
                            break
                    if solapada:
                        continue
                    
                    # Recortar y uardar la nueva imagen
                    imagen_recortada = img.crop((xmin, ymin, xmax, ymax))
                    nuevo_nombre = f'{img_name}_{x}_{y}_{indice_x}_{indice_y}.jpg'
                    imagen_recortada.save(f'./dataset_expandido/{nuevo_nombre}')
                    nuevas_clases = nuevas_clases._append(
                        {'filename': nuevo_nombre, 'Sospechoso': clase}, ignore_index=True)
                    imagenes_guardadas.append((xmin, ymin, xmax, ymax)) 
        bar.display()        



def cargar_modelo():
    # Cargar el modelo YOLOv7 local con los pesos pre-entrenados
    print("Cargando modelo")
    model = torch.hub.load('WongKinYiu/yolov7', 'custom',
                           'pap.pt', source='github')

    # Poner el modelo en modo de evaluación
    model.eval()

    if torch.cuda.is_available():
        print("Usando GPU")
        model.cuda()  # Mueve el modelo a la GPU

    return model


if __name__ == '__main__':
    print("Expansión del dataset")
    expansion_dataset_heatmap(416, 416, umbral_celulas=0)
    print("Dataset expandido")
