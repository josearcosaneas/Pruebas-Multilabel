# -*- coding: utf-8 -*- 
"""
"""
#path="/home/blunt/Escritorio/TFG-SIDP/iniciativas/"
import os
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from xml.dom.minidom import parse
import numpy as np
#print len(stem)
#stem=list(set(stem))
##print len(stem)
#stem = dict.fromkeys(stem)
#stem= stem.keys()
"""
Quitamos espacios para pasar las materias unidas.
"""
def Replace(String):
    String = String.replace(" ","")
    return String
"""
"""
espacio = ' '
def unirPalabras(lista):
    salida = ''
    for i in lista:
        salida += espacio + i
    return salida
"""
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
"""
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
"""
"""    
def tokenizeT(result):
    entradaT=[]
    for i in range(0,len(result)):
        entrada = tokenize(result[i])            
    entradaT.append(entrada)
        
"""
"""
def transformaMaterias(materiasF):
    label=[]
    materiasF=materiasF[0].split(",")

    for i in range (1,len(materiasF)):  

        nuevo=str(materiasF[i])
        nuevo=Replace(nuevo)
        label.append(nuevo)

    return label
"""
"""
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
GeneraTarget(materias)
"""
"""
ficheros = os.listdir('/home/blunt/Escritorio/TFG-SIDP/iniciativas') # linux
materias = []
parrafos = []
extracto = []

for i in ficheros:

    path="/home/blunt/Escritorio/TFG-SIDP/iniciativas/"+i
    
    materias.append(leerTags(path,'materias'))
    parrafos.append(leerTags(path,'parrafo'))
    extracto.append(leerTags(path,'extracto'))
    
    
"""
"""

   
    
def PreparaParrafos(parrafos):
    tokensTrain = []
    for i in parrafos:
        tokens = tokenize(i)
        STOP_TYPES =['.',',',':',';','desde' , 'para', 'por', 'a' , 'ante','bajo','cabe','con','contra','asi','durante','en','hacia','este','mediante','para','por','segun','sin','so','sobre','tras','vresus','via']
        good_words = [w for w in tokens if w not in STOP_TYPES]
        aux=unirPalabras(good_words)# Formamos un texto
        aux = str(aux)
        tokensTrain.append(aux)
    return tokensTrain
"""

"""
def PreparaTest(parrafo):
    #tokensTest=[]
    tokens = tokenize(parrafo)
    STOP_TYPES =['.',',',':',';','desde' , 'para', 'por', 'a' , 'ante','bajo','cabe','con','contra','asi','durante','en','hacia','este','mediante','para','por','segun','sin','so','sobre','tras','vresus','via']
    good_words = [w for w in tokens if w not in STOP_TYPES]
    aux=unirPalabras(good_words)
    aux = str(aux)
    return aux
#PreparaParrafos(parrafos)
"""
"""    
def PreparaMaterias(materias):
    materiasTrain = []
    for i in materias:
        if len(i)>0:
            i=transformaMaterias(i)
        materiasTrain.append(i)
    return materiasTrain
    
#PreparaMaterias(materias)

