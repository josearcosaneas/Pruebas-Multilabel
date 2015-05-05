# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:44:38 2015

@author: blunt
"""

import os
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

ficherosM = os.listdir('/home/blunt/Escritorio/materiasT/') # linux
ficherosT = os.listdir('/home/blunt/Escritorio/iniciativasTxt/') # linux
def creartxt():
    archi=open('datos.txt','w')
    archi.close()
def nula(variable):       # Implementamos la funcion.
  if(len(variable)==0):  # Preguntamos por la longitud de la variable.
    return True             # Si es 0 la funcion devuelve Verdadero(True)
  else:
    return False    
def grabartxt():
    archi=open('datos.txt','a')
    archi.write('Linea 1\n')
    archi.write('Linea 2\n')
    archi.write('Linea 3\n')
    archi.close()
    
def leertxt(path):
    parrafos = []
    archi=open(path,'r')
    linea=archi.readline()
    while linea!="" :
        if len(linea)>1:
            parrafos.append(linea)
        linea=archi.readline()
    archi.close()
    return parrafos
def tokenize(resultList):
    stem=[]
    for i in range(0,len(resultList)):
        sentence=resultList[i]
        tokens = word_tokenize(sentence)
        filtered_words = [w for w in tokens if not w in stopwords.words('spanish')]
        STOP_TYPES =['.',',',':',';','desde' , 'para', 'por', 'a' , 'ante','bajo','con','contra','asi']
        good_words = [w for w in filtered_words if w not in STOP_TYPES]

        stemmer = SnowballStemmer('spanish')
    
        for i in good_words:   
            stem.append( stemmer.stem(i))

materias =[]
for i in ficherosM:
    path="/home/blunt/Escritorio/materiasT/"+i
    materia = leertxt(path)
    materias.append(materia)
    
parrafos = []
for i in ficherosT:
    path="/home/blunt/Escritorio/iniciativasTxt/"+i 
    parrafo = leertxt(path)
    tokenize(parrafo)
    parrafos.append(parrafo)
    
