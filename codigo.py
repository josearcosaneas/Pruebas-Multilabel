# -*- encoding: utf-8 -*-
"""
Created on Sun May 10 15:45:07 2015
@author: blunt
"""
import time

import os
from summa import summarizer
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

def leerTags(path,tag):
    midom=parse(path)
    elements = midom.getElementsByTagName(tag)
    resultList1 = []

    if len(elements) > 0:
        for i in range(0,len(elements)):
            resultList1.extend([elements[i].childNodes[0].nodeValue])
    return resultList1
    
ficherosT = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTest') # linux
ficherosE = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTraining')
materiasE = []
extractoE = []
materiasT = []
extractoT = []

j=0
for i in ficherosT:
    
    path="/home/blunt/Escritorio/TFG-SIDP/iniciativas/"+i
    j=j+1    
    materiasT.append(leerTags(path,'materias'))
    extractoT.append(leerTags(path,'parrafo'))

# Leemos los de entrenamiento
for i in ficherosE:
    path="/home/blunt/Escritorio/TFG-SIDP/iniciativas/"+i
    materiasE.append(leerTags(path,'materias'))
    extractoE.append(leerTags(path,'parrafo'))


for i in range(0,len(extractoE)):
    for j in range(0,len(extractoE[i])):
        extractoE[i][j]= str(extractoE[i][j])
for i in range(0,len(extractoT)):
    for j in range(0,len(extractoT[i])):
        extractoT[i][j]= str(extractoT[i][j])
        
        
def Replace(String):
    String = String.replace(" ","")
    return String    
"""
Genera el tesauro
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

target_names=GeneraTarget(materias)     
def unirLista(listas):
	salida = []
	
	for i in range(0,len(listas)):
		inicio = '' 
		for j in range(0, len(listas[i])):
			inicio = inicio + listas[i][j]
		salida.append(inicio)
	# por cada uno de los elementos de mi nueva lista 
	# lo transformo al tipo de datos de entrada para el clasificador
	for i in range(0,len(salida)):
		salida[i]= np.string_(salida[i])
	return salida

def resumir(texto,lenguaje='spanish'):
    if not(lenguaje):
        return summarizer.summarize(texto, language='spanish')
    else:
        return summarizer.summarize(texto, language=lenguaje)

def resumirTodo(extractoT):
    for i in range(0,len(extractoT)):
        extractoT[i] =str(extractoT[i])
    for i in range(0,len(extractoT)):
        extractoT[i]=resumir(extractoT[i])        
    return extractoT


"""
Funcion encargada de transformar materias para su entrada 
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
Preparacion de materias para pasar al clasificador 
"""    
def PreparaMaterias(materias):
    materiasTrain = []
    for i in materias:
        if len(i)>0:
            i=transformaMaterias(i)
        materiasTrain.append(i)
    
    return materiasTrain

"""
Funcion que se encarga de clasificar:
    X_train - conjunto X para training - type -> np.array y los elementos de este del tipo np.string_
    y_train - conjunto y para training - type -> list de list
    X_test - conjunto X para test - type -> np.array y los elementos de este del tipo np.string_
    target_names - es la lista de todos los elementos del Tesauro - type -> list
"""    
def clasificador(X_train, y_train, X_test, target_names):
    
    lb = preprocessing.MultiLabelBinarizer()
    
    Y = lb.fit_transform(y_train)
    
    classifier = Pipeline([
        ('vectorizer',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf',OneVsRestClassifier(LinearSVC()))])
        
    cv = KFold(Y.shape[0], n_folds=3, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, X_train, y_train, scoring='f1', cv=cv)
    print("CV scores.")
    print(scores)
    print("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    classifier.fit(X_train,Y)
    predicted = classifier.predict(X_test)
    all_label = lb.inverse_transform(predicted)
    print all_label
#    
#    for item, labels in zip(X_test, all_label):
#        print '%s => %s' % (item, ','.join(labels))
    
extractoT= resumirTodo(extractoT)
iniciativasTraining = np.array(unirLista(extractoT))
#
extractoE = resumirTodo(extractoE)
iniciativasTest= np.array(unirLista(extractoE))

y_test = PreparaMaterias(materiasE)
y_train = PreparaMaterias(materiasT)


clasificador(iniciativasTraining, y_train, iniciativasTest, target_names)

