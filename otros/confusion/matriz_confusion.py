# Método para generar la matriz de confusión. 
# Hay que editar el parámetro del TIMESTAMP para poner el del modelo que quieres evaluar.
# El primer bloque de código está copiado del de Ignacio, pero con "INCISO" de vez en cuando, que son necesarios
import os
import json
import numpy as np
import matplotlib.pylab as plt
import seaborn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

from imgclas.data_utils import load_image, load_data_splits, load_class_names
from imgclas.test_utils import predict
from imgclas import paths, plot_utils, utils, test_utils

# User parameters to set
TIMESTAMP = input("Indica el timestamp. Sin espacios. Mismo formato que en models: ")                       # timestamp of the model
MODEL_NAME = 'final_model.h5'                           # model to use to make the prediction
TOP_K = 2                                               # number of top classes predictions to save

# Set the timestamp
paths.timestamp = TIMESTAMP

# Load the data
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++antes de class names")
class_names = load_class_names(splits_dir=paths.get_ts_splits_dir()) # INCISO: Estas son las clases que había en el modelo
# en el momento en el que estrenaste (dado por el timestamp). No las que tienes en data/dataset_files
print("----------------------------------------------------------------------despues de class names")
# Load training configuration
conf_path = os.path.join(paths.get_conf_dir(), 'conf.json')
with open(conf_path) as f:
    conf = json.load(f)
    
# Load the model
print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------antes")
model = load_model(os.path.join(paths.get_checkpoints_dir(), MODEL_NAME))
#model = load_model(os.path.join(paths.get_checkpoints_dir(), MODEL_NAME), custom_objects=utils.get_custom_objects())
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++después")
# INCISO: Ahora la parte que continúa está basada en el predicting a datasplit txt file que incluye Ignacio en el notebook
# 3.0 . Esta preparación previa es necesaria para computar la matriz de confusión. 
#
# OJO: ahora lo que le vas a dar para testear el modelo dado por el timestamp SÍ se encuentra en data/dataset_files 
# Y ES CON LO QUE TÚ QUIERES TESTEAR EL MODELO.
SPLIT_NAME = input("Indica el nombre del split con el que evaluas. Es de data/dataset_files. Ejemplos: val train ...: ")
# Load the data
X, y = load_data_splits(splits_dir=paths.get_ts_splits_dir(),
                        im_dir=conf['general']['images_directory'],
                        split_name=SPLIT_NAME)
# Predict
# Añade esto si quieres no usar aumentacion en la validacion:
# 
print(conf['augmentation']['val_mode'])
pred_lab, pred_prob = predict(model, X, conf, top_K=TOP_K, filemode='local')

#Ahora guardamos las predicciones
# Save the predictions
pred_dict = {'filenames': list(X),
             'pred_lab': pred_lab.tolist(),
             'pred_prob': pred_prob.tolist()}
if y is not None:
    pred_dict['true_lab'] = y.tolist()
# No incluimos la parte de guardarlo en json porque lo vamos a utilizar ahora mismo. 
# Importamos el warning este porque Ignacio lo sugiere.
import warnings
warnings.filterwarnings("ignore") # To ignore UndefinedMetricWarning: [Recall/Precision/F-Score] is ill-defined and being set to 0.0 in labels with no [true/predicted] samples.

# INCISO: Sacamos por pantalla distintas métricas relevantes SOBRE EL SPLIT SELECCIONADO

true_lab, pred_lab = np.array(pred_dict['true_lab']), np.array(pred_dict['pred_lab'])

top1 = test_utils.topK_accuracy(true_lab, pred_lab, K=1)
top2 = test_utils.topK_accuracy(true_lab, pred_lab, K=2)
# INCISO: LO COMENTO PORQUE QUIERO EL TOP 2 top5 = test_utils.topK_accuracy(true_lab, pred_lab, K=5)

# INCISO: También vamos a guardarlo en un .txt, del que solicitaremos nombre al usuario
nombre_metricas = input("Ponle nombre al fichero con las métricas relevantes. No especifiques formato, va a ser .txt por defecto: " )
nombre_metricas = nombre_metricas + ".txt"
m = open(nombre_metricas,'w')

print('Top1 accuracy: {:.1f} %'.format(top1 * 100))
m.write('Top1 accuracy: {:.1f} %'.format(top1 * 100) + '\n')
print('Top2 accuracy: {:.1f} %'.format(top2 * 100))
m.write('Top2 accuracy: {:.1f} %'.format(top2 * 100) + '\n')
# INCISO ESTO LO COMENTO PORQUE AHORA QUIERO TOP 2 print('Top5 accuracy: {:.1f} %'.format(top5 * 100))
#m.write('Top5 accuracy: {:.1f} %'.format(top5 * 100) + '\n')

labels = range(len(class_names))

print('\n')
m.write('\n')

print('Micro recall: {:.1f} %'.format(100 * recall_score(true_lab, pred_lab[:, 0], labels=labels, average='micro')))
m.write('Micro recall: {:.1f} %'.format(100 * recall_score(true_lab, pred_lab[:, 0], labels=labels, average='micro')) + '\n')

print('Macro recall: {:.1f} %'.format(100 * recall_score(true_lab, pred_lab[:, 0], labels=labels, average='macro')))
m.write('Macro recall: {:.1f} %'.format(100 * recall_score(true_lab, pred_lab[:, 0], labels=labels, average='macro')) + '\n')

print('Macro recall (no labels): {:.1f} %'.format(100 * recall_score(true_lab, pred_lab[:, 0], average='macro')))
m.write('Macro recall (no labels): {:.1f} %'.format(100 * recall_score(true_lab, pred_lab[:, 0], average='macro')) + '\n')

print('Weighted recall: {:.1f} %'.format(100 * recall_score(true_lab, pred_lab[:, 0], labels=labels, average='weighted')))
m.write('Weighted recall: {:.1f} %'.format(100 * recall_score(true_lab, pred_lab[:, 0], labels=labels, average='weighted')) + '\n')

print('\n')
m.write('\n')

print('Micro precision: {:.1f} %'.format(100 * precision_score(true_lab, pred_lab[:, 0], labels=labels, average='micro')))
m.write('Micro precision: {:.1f} %'.format(100 * precision_score(true_lab, pred_lab[:, 0], labels=labels, average='micro')) + '\n')

print('Macro precision: {:.1f} %'.format(100 * precision_score(true_lab, pred_lab[:, 0], labels=labels, average='macro')))
m.write('Macro precision: {:.1f} %'.format(100 * precision_score(true_lab, pred_lab[:, 0], labels=labels, average='macro')) + '\n')

print('Macro precision (no labels): {:.1f} %'.format(100 * precision_score(true_lab, pred_lab[:, 0], average='macro')))
m.write('Macro precision (no labels): {:.1f} %'.format(100 * precision_score(true_lab, pred_lab[:, 0], average='macro')) + '\n')

print('Weighted precision: {:.1f} %'.format(100 * precision_score(true_lab, pred_lab[:, 0], labels=labels, average='weighted')))
m.write('Weighted precision: {:.1f} %'.format(100 * precision_score(true_lab, pred_lab[:, 0], labels=labels, average='weighted')) + '\n')

print('\n')
m.write('\n')

print('Micro F1 score: {:.1f} %'.format(100 * f1_score(true_lab, pred_lab[:, 0], labels=labels, average='micro')))
m.write('Micro F1 score: {:.1f} %'.format(100 * f1_score(true_lab, pred_lab[:, 0], labels=labels, average='micro')) + '\n')

print('Macro F1 score: {:.1f} %'.format(100 * f1_score(true_lab, pred_lab[:, 0], labels=labels, average='macro')))
m.write('Macro F1 score: {:.1f} %'.format(100 * f1_score(true_lab, pred_lab[:, 0], labels=labels, average='macro')) + '\n')

print('Macro F1 score (no labels): {:.1f} %'.format(100 * f1_score(true_lab, pred_lab[:, 0], average='macro')))
m.write('Macro F1 score (no labels): {:.1f} %'.format(100 * f1_score(true_lab, pred_lab[:, 0], average='macro')) + '\n')

print('Weighted F1 score: {:.1f} %'.format(100 * f1_score(true_lab, pred_lab[:, 0], labels=labels, average='weighted')))
m.write('Weighted F1 score: {:.1f} %'.format(100 * f1_score(true_lab, pred_lab[:, 0], labels=labels, average='weighted')) + '\n')

m.close()

# INCISO: YA VAMOS CON LA MATRIZ DE CONFUSIÓN!!
def plt_conf_matrix(conf_mat, labels=False):
    
    fig = plt.figure(figsize=(20, 20))
    hm = seaborn.heatmap(conf_mat, annot=False, square=True, cbar_kws={'fraction':0.046, 'pad':0.04},
                         xticklabels=labels, yticklabels=labels, cmap="YlGnBu")
    fontsize = None
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=fontsize)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
y_true, y_pred = np.array(pred_dict['true_lab']), np.array(pred_dict['pred_lab'])[:, 0] #Aqui, al poner [:,0] te asegura que de todas las probabilidades
# tú coges la que más tienes como etiqueta predicha
conf_mat = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), sample_weight=None)
normed_conf = conf_mat / np.sum(conf_mat, axis=1)[:, np.newaxis]

# plt_conf_matrix(conf_mat)
plt_conf_matrix(normed_conf, labels=class_names)
nombre_confusion = input("Ponle nombre a la matriz de confusion. No especifiques formato: " )
plt.savefig(nombre_confusion,dpi = 50,format = "png", bbox_inches = 'tight')
