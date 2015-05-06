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
Quitamos espacios para pasar las materias unidas.
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
ficherosT = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTraining') # linux
ficherosE = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTest')
materiasE = []
extractoE = []
materiasT = []
extractoT = []
for i in ficherosT:
    path="/home/blunt/Escritorio/TFG-SIDP/iniciativas/"+i
    materiasT.append(leerTags(path,'materias'))
    extractoT.append(leerTags(path,'extracto'))

    
for i in ficherosE:
    path="/home/blunt/Escritorio/TFG-SIDP/iniciativas/"+i
    materiasE.append(leerTags(path,'materias'))
    extractoE.append(leerTags(path,'extracto'))
    
"""
"""
def Replace(String):
    String = String.replace(" ","")
    return String
"""
"""
    

def clasificador(X_train, y_train, X_test, target_names):
    
    lb = preprocessing.LabelBinarizer()
    
    Y = lb.fit_transform(y_train)
    
    classifier = Pipeline([
        ('vectorizer',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf',OneVsRestClassifier(LinearSVC()))])
        
#    cv = KFold(y_train.shape[0], n_folds=3, shuffle=True, random_state=42)
#    scores = cross_val_score(classifier, X_train, y_train, scoring='f1'"""roc_auc""", cv=cv)
#    print("CV scores.")
#    print(scores)
#    print("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    classifier.fit(X_train,Y)
    predicted = classifier.predict(X_test)
    all_label = lb.inverse_transform(predicted)
    
    for item, labels in zip(X_test, all_label):
        print '%s => %s' % (item, ','.join(labels))



def evalucion(Y_predict, Y_test, etiqueta):
    if etiqueta == 'Precision':
        print '--> Precision = '+ precision_score(Y_predict, Y_test)
    elif etiqueta == 'Recall':
        print '--> Recall= '+recall_score(Y_predict,Y_test)
    elif etiqueta == 'F1-Score':
        print '--> F1-score= '+f1_score(Y_predict,Y_test)
    else:
        print '--> Precision = '+ precision_score(Y_predict, Y_test) 
        print '--> Recall= '+ recall_score(Y_predict,Y_test)
        print '--> F1-score= '+ f1_score(Y_predict,Y_test)
        


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
    #tokens = tokenize(parrafo)
      
    STOP_TYPES =['.',',',':',';','desde' , 'para', 'por', 'a' , 'ante','bajo','cabe','con','contra','asi','durante','en','hacia','este','mediante','para','por','segun','sin','so','sobre','tras','vresus','via']
    parrafo =np.array(parrafo)    
    good_words = [w for w in parrafo if w not in STOP_TYPES]
    #aux=unirPalabras(good_words)
    aux = np.array(good_words)
    
    return aux

"""
"""    
def PreparaMaterias(materias):
    materiasTrain = []
    for i in materias:
        if len(i)>0:
            i=transformaMaterias(i)
        materiasTrain.append(i)
    
    return materiasTrain
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
    
#PreparaMaterias(materiasT)
X_train = PreparaTest(extractoE)
y_train = PreparaMaterias(materiasE)

X_test = PreparaTest(extractoT)
y_test = PreparaMaterias(materiasT)
print len(X_train)
r = 0
entrena = []
for i in range(0,len(X_train)):
    entrena.append(X_train[i][0])

for i in range (0,len(entrena)):
    entrena [i]=np.string_(entrena[i])
 
test = []
for i in range(0,len(X_test)):
    test.append(X_test[i][0])

for i in range (0,len(test)):
    test[i]=np.string_(test[i])





#print extractoT[0][0]


print type(X_train[0])
print type(y_train[0])
target_names=GeneraTarget(materias)
print type(target_names[0])
print type(target_names)
print type(entrena[0])
print type(test[0])


#print X_train
#print y_train
#print test
#print entrena
clasificador(entrena, y_train, test, target_names)
