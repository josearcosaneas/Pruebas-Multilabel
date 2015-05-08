# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:57:01 2015

@author: blunt
"""

# -*- coding: utf-8 -*- 
"""
Created on Tue May  5 22:44:00 2015

@author: Jose Arcos Aneas

    Archivo que se encarga de clasificar los datos y evaluar los resultados
    obtenidos por el clasificador.
    
"""

import os
from nltk.corpus import stopwords
from summa import summarizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from xml.dom.minidom import parse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cross_validation import cross_val_score, KFold

#print len(stem)
#stem=list(set(stem))
##print len(stem)
#stem = dict.fromkeys(stem)
#stem= stem.keys()
"""
Funcion encargada de Leer un Tags en un archivo XML.
    Path = lugar de almacenamiento del fichero.
    Tags = tag que vamos leer.
"""

def leerTags(path,tag):
    midom=parse(path)
    elements = midom.getElementsByTagName(tag)
    resultList1 = []
    if len(elements) > 0:
        for i in range(0,len(elements)):
            resultList1.extend([elements[i].childNodes[0].nodeValue])
    return resultList1
"""
Leemos los directorios para recorrerlos y leer todos los archivos de ellos
"""
ficherosT = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTest') # linux
extractoT = []
ficherosE = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTraining')
extractoE = [] 

# leemos todos los archivos de los ficheros en los que estamos interasados
# leemos los archivos de training
for i in ficherosT:
    path="/home/blunt/Escritorio/iniciativas/iniciativasTest/"+i
    extractoT.append(leerTags(path,'parrafo'))
for i in ficherosE:
    path="/home/blunt/Escritorio/iniciativas/iniciativasTraining/"+i
    extractoE.append(leerTags(path,'parrafo'))
    
    

"""
Funcion que se encarga de resumir un texto
"""

def resumir(texto,lenguaje='spanish'):
    if not(lenguaje):
        return summarizer.summarize(texto, language='spanish')
    else:
        return summarizer.summarize(texto, language=lenguaje)

    
def resumirTodo(parrafos):

    for i in range(0,len(extractoT)):
        extractoT[i] =str(extractoT[i])
    for i in range(0,len(extractoT)):
        print len (extractoE[i])
        extractoT[i]=str(resumir(extractoT[i]))
        print extractoT[i]
    return extractoT


print type(extractoE[0])
extractoT = resumirTodo(extractoT)


print extractoT

#print type(extractoE[0])
#
#print type(extractoE)
#print type(extractoT[0])
#
#print type(extractoT)

#print X_train[0]