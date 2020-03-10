#vamos a coger los paths con etiqueta, hacer un pequeño barajeo y asignar arbitrariamente un 30%
#a la parte de test, con el 70% restante a entrenamiento. Para ello nos ayudaremos de la
#librería de data science sklearn
#Toma como ENTRADA el train generado con el 100% de las imagenes con organizador.py
#Como salidas da un train con un 70% del train original y un val con el 30% restante
import numpy as np
from sklearn.model_selection import train_test_split
#Con esto generamos las entradas del metodo de sklearn:
#
tp = open('train_part.txt','w')#Creo el fichero donde estará el train parcial
v = open('val.txt','w')#Creo el fichero con los datos de validacion
t = open('train.txt','r') #Declaro el fichero que voy a leer. En el docker será el train.txt 
#original creado por el organizador.py. Aquí usamos el fichero de muestra toy.txt
almacena_paths = np.array([])
etiquetas = []
for line in t:
    almacena_paths = np.append(almacena_paths,line.split()[0])
    etiquetas.append(line.split()[1])
#Ya tenemos las etiquetas por un lado y los paths por otro, así que estamos listos
#Para preparar los datos con sklearn. Después los escribiremos en dos ficheros llamados
#Train_part y val Hace shuffle por defecto Shuffle = True
X_train, X_test, y_train, y_test = train_test_split(almacena_paths, etiquetas, test_size=0.10, random_state=42)
for contador in range(len(y_train)):
    tp.write(X_train[contador] + ' ' + y_train[contador] + '\n')
for contador_val in range(len(y_test)):
    v.write(X_test[contador_val] + ' ' + y_test[contador_val] + '\n')
