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
ficherosT = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTraining') # linux
extractoT = []

# leemos todos los archivos de los ficheros en los que estamos interasados
# leemos los archivos de training
for i in ficherosT:
    path="/home/blunt/Escritorio/iniciativas/iniciativasTraining/"+i

    extractoT.append(leerTags(path,'parrafo'))
    
    
"""
Funcion que se encarga suprimir espacios
"""
def Replace(String):
    String = String.replace(" ","")
    return String
"""
Funcion que se encarga de resumir un texto
"""

def resumir(texto,lenguaje='spanish'):
    if not(lenguaje):
        return summarizer.summarize(texto, language='spanish')
    else:
        return summarizer.summarize(texto, language=lenguaje)
espacio = ' '
def unirPalabras(lista):
    salida = ''
    for i in lista:
        salida += espacio + i
    return salida
def tokenize(resultList1):
    entrada=[]
    for i in range(0,len(resultList1)):
        sentence=resultList1[i]
        tokens = word_tokenize(sentence)
        filtered_words = [w for w in tokens if not w in stopwords.words('spanish')]

        stemmer = SnowballStemmer('spanish')
        for i in filtered_words:
            entrada.append( stemmer.stem(i))
    return entrada
        
def PreparaParrafos(parrafos):
    tokensTrain = []
    for i in parrafos:
        print i[0]
#        tokens = tokenize(i)
#        print tokens
#        STOP_TYPES =['.',',',':',';','desde' , 'para', 'por', 'a' , 'ante','bajo','cabe','con','contra','asi','durante','en','hacia','este','mediante','para','por','segun','sin','so','sobre','tras','vresus','via']
#        good_words = [w for w in tokens if w not in STOP_TYPES]
#        aux=unirPalabras(good_words)# Formamos un texto
#        aux = aux
#        tokensTrain.append(aux)
       
    return tokensTrain

#print extractoT
#print PreparaParrafos(extractoT)

    
def resumirTodo(parrafos):
    for i in range(0,len(extractoT)):
        extractoT[i] =str(extractoT[i])
    for i in range(0,len(extractoT)):
        extractoT[i]=resumir(extractoT[i])
    return extractoT
extractoT = resumirTodo(extractoT)
print extractoT[0]
X_train = PreparaParrafos(extractoT)
#print X_train[0]