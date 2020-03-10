# Script para aplicar a train/val/test.txt. Añade, tras la etiqueta de clases, una nueva de localización. Esto se guarda en un nuevo fichero con el mismo nombre mas un "con_locations". También genera un fichero análogo a classes. 
# a modo de leyenda de localizaciones, llamado locations.txt
# @Author Luis Moro Carrera
with open("locations.txt", "w") as l:
    l.write("EASTERN_ATLANTIC" +  "\n") # Etiqueta 0
    l.write("EASTERN_PACIFIC" +  "\n") # Etiqueta 1
    l.write("INDO-PACIFIC" +  "\n") # Etiqueta 2
    l.write("MEDITERRANEAN" +  "\n") # Etiqueta 3
    l.write("WESTERN_ATLANTIC" +  "\n") # Etiqueta 4
    l.close()
fichero = input("Escoge sobre qué fichero aplicar esto train/val/text, sin .txt : ")
fichero_nuevo = fichero + "_con_location" + ".txt" 
fichero = fichero + ".txt"

def retorna_location(path):
    if "EASTERN_ATLANTIC" in path:
        return str(0)
    if "EASTERN_PACIFIC" in path:
        return str(1)
    if "INDO-PACIFIC" in path:
        return str(2)
    if "MEDITERRANEAN" in path:
        return str(3)
    if "WESTERN_ATLANTIC" in path:
        return str(4)
localizaciones_lista = [] # genero lista vacía
with open(fichero, "r") as f:
    for line in f:    
        localizaciones_lista.append(line[:-1] + " " + retorna_location(line) + "\n")
f.close()
# Hago este paso intermedio de guardarlo en una lista porque no quiero correr el riesgo de sobrescribir el original con el with open
with open(fichero_nuevo, "w") as n:
    for line in localizaciones_lista:
        n.write(line)
n.close()


