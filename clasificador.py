# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:44:00 2015

@author: Jose Arcos Aneas

    Archivo que se encarga de clasificar los datos y evaluar los resultados
    obtenidos por el clasificador.
    
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cross_validation import cross_val_score, KFold


    

def clasificador(X_train, y_train, X_test, target_names):
    
    lb = preprocessing.LabelBinarizer()
    
    Y = lb.fit_transform(y_train)
    
    classifier = Pipeline([
        ('vectorizer',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf',OneVsRestClassifier(LinearSVC()))])
        
    cv = KFold(y_train.shape[0], n_folds=3, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, X_train, y_train, scoring='f1'"""roc_auc""", cv=cv)
    print("CV scores.")
    print(scores)
    print("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
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
        


X_test = np.array(['nice day in nyc',
                   'welcome to london',
                   'london is rainy',
                   'it is raining in britian',
                   'it is raining in britian and the big apple',
                   'it is raining in britian and nyc',
                   'hello welcome to new york. enjoy it here and london too'])
print type(X_test)
        
y_train_text = [["new york"],["new york"],["new york"],["new york"],["new york"],
                ["new york"],["london"],["london"],["london"],["london"],
                ["london"],["london"],["new york","london"],["new york","london"]]

print type(y_train_text)
    
    
