# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:44:00 2015

@author: Jose Arcos Aneas

    Archivo que se encarga de clasificar los datos y evaluar los resultados
    obtenidos por el clasificador.
    
"""
import re
from xml.dom.minidom import parse
import os
import string


def Replace(String):
    String = String.replace(" ","")
    return String

# Lectura de ficheros del directorio
ficheros = os.listdir('/home/blunt/Escritorio/TFG-SIDP/iniciativas') # linux
materias = []
# Extraccion de las materias por cada iniciativa del directorio
# las termino uniendo en una lista.
for i in ficheros:
    path="/home/blunt/Escritorio/TFG-SIDP/iniciativas/"+i
    midom=parse(path)    
    elements = midom.getElementsByTagName('materias')
    resultList = []
    if len(elements) != 0:
        for i in range(0,len(elements)):
            resultList.extend([elements[i].childNodes[0].nodeValue])
            materias.append(resultList[i])
#for i in range(0,len(materias)):
#    materias[i] = materias[i].split(",")      
#materiasF=materias[0]
#for i in range(1,len(materias)):
#    for j in range(0,len(materias[i])):
#        if materias[i][j] not in materiasF:
#            materiasF.append(materias[i][j])
#            
#label=[]
#for i in range (1,len(materiasF)):    
#    nuevo=str(materiasF[i])
#    nuevo=Replace(nuevo)
#    #print nuevo
#    label.append(nuevo)
#label=list(set(label))
#print label
    
    
def GeneraTarget(materias):
    for i in range(0,len(materias)):
        if len(materias[i])>0:
            materias[i] = materias[i].split(",")      
    materiasF=materias[0]
    for i in range(1,len(materias)):
        for j in range(0,len(materias[i])):
            if materias[i][j] not in materiasF:
                materiasF.append(materias[i][j])
    label=[]
    for i in range (1,len(materiasF)):    
        nuevo=str(materiasF[i])
        nuevo=Replace(nuevo)
        #print nuevo
        label.append(nuevo)
    label=list(set(label))
    return label
    
target_set=GeneraTarget(materias)
print target_set
    
    
    
    