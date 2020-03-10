#Importamos todo lo necesario
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.image as mim
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os
from matplotlib import pyplot as plt
dimension_reescalar = (224, 224) #Las dimensiones (para cada uno de los 3 rgb que queremos en cada foto)
path_del_val = '/srv/image-classification-tf/data/dataset_files/val.txt'
path_del_modelo = '/srv/image-classification-tf/models/2020-02-04_182818/ckpts/final_model.h5' 
modelo = load_model(path_del_modelo ,compile=False) #Importas el modelo que quieras .h5
vector_entradas = np.array(([])) #inicializo vector vacío
etiquetas_reales = np.array(([])) #Donde guardaremos las etiquetas reales para la matriz de confusión
contador = 0 #tengo que saber cuántas imágenes voy a hacer ya que numpy te guarda todos los coeficientes en una fila
#y, antes de pasarle a la red las imágenes, te toca hacer un resize a (numero_de_fotos,224,224,3)

#EJEMPLO BIEN HECHO CON LARA PARA UNA SOLA IMAGEN:
path_imagen = '/srv/image-classification-tf/data/images/Conus/INDO-PACIFIC/lohri_IP/006-lohri-IP-Mozambique.jpg' #categoría 31
imagen_sola = image.load_img(path_imagen, target_size=(224,224))
imagen_sola = np.expand_dims(imagen_sola, axis = 0)
prediccion_sola = modelo.predict(imagen_sola)
print(prediccion_sola)




with open(path_del_val, 'r') as path:
    for x in path:
        print(x)
        etiquetas_reales = np.append(etiquetas_reales, x.split()[1])
        #pasamos la imagen al numpy
        imagen = image.load_img(x.split()[0], target_size = dimension_reescalar)
        imagen = image.img_to_array(imagen)
        imagen = np.expand_dims(imagen, axis = 0)
        vector_entradas = np.append(vector_entradas, imagen)
        contador = contador + 1
#Ya tenemos en una sola fila todos los coeficientes de todas las imágenes. Vamos a hacer el reshape para tenerlas como las requiere la red
#Es decir (numero_de_fotos, 224,224,3)
print(vector_entradas.shape)
vector_entradas = np.resize(vector_entradas,(contador,224,224,3))
print("Y después")
print(vector_entradas.shape)
#vector_parcial = vector_entradas[0] PRUEBAS
#vector_parcial = np.resize(vector_parcial, (1,224,224,3)) PRUEBAS
predicciones = modelo.predict(vector_entradas) #le pasamos en un batch todo y nos retorna las predicciones
#cv2.imwrite('foto_comprobar.jpg', vector_entradas[0]) PRUEBAS
print(predicciones[0])
predicciones_sin_densidad = [] #Inicializo LISTA con las predicciones
for i in predicciones:
    clase_mas_probable, = np.where(i == max(i))
    predicciones_sin_densidad.append(clase_mas_probable[0])
print('las predicciones:')
print(predicciones_sin_densidad)
print('la realidad:')
print(etiquetas_reales)

#Como nos retorna densidades de probabilidad, vamos uno por uno viendo para qué clase se lleva la máxima densidad de probabilidad y guardandolo en 
#vector

print("hemos terminado")
