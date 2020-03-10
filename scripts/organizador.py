#Programa de python destinado a generar dos salidas
#ES NECESARIO PONER LAS CARPETAS ANTES DE PROCESARLAS CON UN FORMATO SIN ESPACIO
#PUEDES SUSTITUIR LOS ESPACIOS POR GUIONES BAJOS USANDO EN EL BASH ESTE COMANDO
#for file in conus/* ; do mv "$file" `echo $file | tr ' ' '_'` ; done #Convierto hasta la primera capa de profundidad los espacios
#en _
#for file in conus/*/* ; do mv "$file" `echo $file | tr ' ' '_'` ; done 
#for file in conus/*/*/* ; do mv "$file" `echo $file | tr ' ' '_'` ; done #y así hasta llegar al final
#Si lo haces con muchas capas al principio, al ser recursivo no reconoce los path, pues ya no existen al haberlos modificado en capas
#superiores, así que sigue el proceso indicado 
#Debe ejecutarse en el mismo fichero que la carpeta principal de las especies a analizar. Es decir, desde donde empieza a dividirse (i.e "Conus","Flores"
#@Author :Luis Moro Carrera
##########Fichero classes.txt: Establece las clases de especies a analizar. La primera clase  (primera fila)
###########################se corresponde con el tag 0 en el fichero train.txt. Luego todas las fotos que tras el path tengan un
###########################número n en el train.txt serán fotografías de la especie de animal escrita en la fila n-1 esima de classes.txt
##########################(primera fila de classes-->tag 0 en train.txt y así sucesivamente
##########################
##########Fichero train.txt: los paths de las imagenes con el tag que los vincula con una especie de las escritas en classes.txt
####################################################################################################################################
import os
t = open('train.txt','w') #Creo el fichero train
t.close() #Lo cierro de momento
c = open('classes.txt','w') #Creo la clase
c.close()
l = open('location.txt','w') #Creo la localización (la primera localizacion es la de la primera clase y asi sucesivamente)
l.close()
numerador_especie = -1 #Inicializo la variable para asignar los tags a las especies
raiz = os.path.abspath('.') #El directorio que actuará de raiz

def directorio_no_esta_vacio(path_del_dir): 
#Clase para comprobar si un directorio está vacio
    comprobador = False
    if len(os.listdir(path_del_dir)) == 0:
        comprobador = False
    else:
        comprobador = True
    return comprobador

def contiene_directorio(path_del_directorio): #definimos funcion que comprueba si dentro de un directorio hay a su vez otro directorio
#o no. Nos interesa saber esto para enterarnos de cuando estamos en un divisor entre distintas subespecies
#o cuando estamos ya en una carpeta que incluye fotografias propiamente. También retorna la lista con los nombres de dichos directorios
    lista_directorios = [] #inicializo lista vacia
    hay_directorio = False #inicializo respuesta por defecto
    for elemento in os.listdir(path_del_directorio):
        if os.path.isdir(os.path.join(path_del_directorio,elemento)):
            hay_directorio = True
            lista_directorios.append(os.path.join(path_del_directorio,elemento))
    return hay_directorio, lista_directorios

#Ya tenemos una forma de comprobar si aún estamos en un divisor entre subespecies que se van ramificando o en una subespecie
#Ya concreta de la que tenemos las fotos
for elemento in os.listdir(raiz):
    subruta = os.path.join(raiz,elemento)
    if os.path.isdir(subruta):
        if contiene_directorio(subruta)[0]:
            for elemento1 in contiene_directorio(subruta)[1]:
                if contiene_directorio(elemento1)[0]:
                    for elemento2 in contiene_directorio(elemento1)[1]:
                        if contiene_directorio(elemento2)[0]:
                            for elemento3 in contiene_directorio(elemento2)[1]:
                                if contiene_directorio(elemento3)[0]:
                                    print('Se ha llegado a la profundidad maxima de ramificaciones que soporta el programa, aún necesitas añadir más for')
                                else:
                                    if directorio_no_esta_vacio(elemento3):
                                        c = open('classes.txt', 'a')
                                        c.write(os.path.basename(elemento3).split('_')[0] + '\n')
                                        c.close()
                                        l = open('location.txt', 'a')                                    
                                        l.write(os.path.basename(elemento3).split('_')[1] + '\n')
                                        l.close()
                                        numerador_especie = numerador_especie + 1
                                        for fotografia3 in os.listdir(elemento3):
                                            t = open('train.txt', 'a')
                                            t.write(os.path.join(elemento3,fotografia3) + ' ' + str(numerador_especie) + '\n')
                                            t.close()
                        else:
                            if directorio_no_esta_vacio(elemento2):
                                c = open('classes.txt', 'a')
                                c.write(os.path.basename(elemento2).split('_')[0] + '\n')
                                c.close()
                                l = open('location.txt', 'a')   
                                l.write(os.path.basename(elemento2).split('_')[1] + '\n')
                                l.close()
                                numerador_especie = numerador_especie + 1
                                for fotografia2 in os.listdir(elemento2):
                                    t = open('train.txt','a')
                                    t.write(os.path.join(elemento2,fotografia2) + ' ' + str(numerador_especie) + '\n')
                                    t.close() 
                else:
                    if directorio_no_esta_vacio(elemento1):
                        c = open('classes.txt', 'a')
                        c.write(os.path.basename(elemento1).split('_')[0] + '\n')
                        c.close()
                        l = open('location.txt', 'a')                                    
                        l.write(os.path.basename(elemento1).split('_')[1] + '\n')
                        l.close()
                        numerador_especie = numerador_especie + 1
                        for fotografia1 in os.listdir(elemento1):
                            t = open('train.txt','a')
                            t.write(os.path.join(elemento1,fotografia1) + ' ' + str(numerador_especie) + '\n')
                            t.close()
        else:
            if directorio_no_esta_vacio(elemento):
                c = open('classes.txt','a')
                c.write(os.path.basename(elemento).split('_')[0] + '\n')
                c.close()
                l = open('location.txt','a')
                print(elemento)
                l.write(os.path.basename(elemento).split('_')[1] + '\n')
                l.close()
                numerador_especie = numerador_especie + 1
                for fotografia in os.listdir(subruta):
                    t = open('train.txt','a')
                    t.write(os.path.join(subruta,fotografia) + ' ' + str(numerador_especie) + '\n')
                    t.close()

