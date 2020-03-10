import matplotlib.pyplot as plt
path_clases = '/srv/image-classification-tf/data/dataset_files/classes.txt' #introduce el path al classes.txt
path_fichero = '/srv/image-classification-tf/data/dataset_files/train.txt' #introduce el path al histograma
d_clases = {} #creo el diccionario donde voy a añadir las clases con su número de etiqueta
contador = 0 #El contador de los índices
with open(path_clases, 'r') as c:
    for i in c:
        d_clases.update(({i.split()[0]:contador}))
        contador = contador+1
print (d_clases)
#Ya tengo el diccionario con las clases, ahora recorremos el fichero del que quiero hacer histograma
#Y voy almacenando en una lista cuántas clases hay de cada, para pasarselo a la funcion de histograma de matplotlyb
#
#necesito diccionario inverso
reverse_d = dict([(int(value), key) for (key, value) in d_clases.items()])
lista_clases = [] #lista donde almaceno las clases de cada muestra
with open(path_fichero, 'r') as f:
    for j in f:
        lista_clases.append(reverse_d[int(j.split()[1])])
#print(lista_clases)
#/srv/image-classification-tf/enlace_fuera/histogramas
histograma = plt.hist(lista_clases)
plt.show()
plt.savefig('/srv/image-classification-tf/enlace_fuera/histogramas/histograma_train.png')
