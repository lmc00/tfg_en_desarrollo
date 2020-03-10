#Importamos todo lo necesario como en el jupyter 1.0 de Ignacio
import os

import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

import imgclas
from imgclas import paths, config
from imgclas.data_utils import load_image, load_data_splits, augment, load_class_names

#Comenzamos a preparar todos los datos

CONF = config.get_conf_dict() #El diccionario con toda la configuracion del yaml
splits_dir = paths.get_splits_dir() #base+data+dataset_files
# Load the training data
X_train, y_train = load_data_splits(splits_dir=splits_dir,
                                    im_dir=CONF['general']['images_directory'],
                                    split_name='train')

# Load the validation data
if (CONF['training']['use_validation']) and ('val.txt' in os.listdir(splits_dir)):
    X_val, y_val = load_data_splits(splits_dir=splits_dir,
                                    im_dir=CONF['general']['images_directory'],
                                    split_name='val')
#load_data_splits comprueba que exista el fichero que se le pasa (ya sean train,val etc). luego con numpy.genfromtxt
#obtiene un array donde la primera columna son los path, en la segunda las etiquetas
#por ultimo retorna un array de numpy con los path absolutos a las fotografias de train o el que le hayas pasado
#y otro con las etiquetas en formato int32 para saber de qué clase son
else:
    print('No validation data.')
    X_val, y_val = None, None
    CONF['training']['use_validation'] = False
    
# Load the class names
class_names = load_class_names(splits_dir=splits_dir)

#Ya tenemos preparado lo básico, así que ahora vamos a calcular la 
#distribución de clases

#Defino algunos parámetros que puedes modificar si te propones añadir los nombres de las etiquetas. 
#Por ejemplo, si quieres que aparezca el nombre de cada clase debajo del bin. Aunque vas a tener que cambiat
#El figsize porque no te van a entrar sino 83 nombres tan apretados.
log_scale = False
show_names = True

# Plot the histograms
 #genero los subplots vacios figsize = (lo ancho, lo alto)

def plot_hist(y, set_name=''):
    fig, ax = plt.subplots(1, figsize=(16,8)) #Genero el subplot vacio
    #figsize = (lo ancho, lo alto)
    n, bins, patches = ax.hist(y, bins=len(class_names), log=log_scale)
    mean, med = np.mean(n), np.median(n)
    ax.axhline(mean, linestyle= '--', color='#ce9b3b', label='mean')
    ax.axhline(med, linestyle= '--', color='#fc0c70', label='median')
    ax.set_title('{} set'.format(set_name))
    ax.legend()
    if show_names:
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation='vertical')

    print('Max {}: {}'.format(set_name, np.amax(n)))
    print('Min {}: {}'.format(set_name, np.amin(n)))
    print('Mean {}: {}'.format(set_name, mean))
    print('Median {}: {}'.format(set_name, med))
    print('\n')
    #Guardamos el plot de la sesión de canvas con el nombre que quiera el usuario
    nombre_hist = input("Ponle nombre al histograma de " + set_name + " :" )
    plt.savefig(nombre_hist,dpi = 100,format = "png")
plot_hist(y_train, set_name='Training')

if y_val is not None:
    plot_hist(y_val, set_name='Validation')

