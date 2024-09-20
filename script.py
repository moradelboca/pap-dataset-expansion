import torch
from PIL import Image
import numpy as np
import os
import pandas as pd
from progress_bar import ProgressBar


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


def cargar_modelo():
    # Cargar el modelo YOLOv7 local con los pesos pre-entrenados
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
    expansion_dataset_divisiones_uniformes(5)
    print("Dataset expandido")
