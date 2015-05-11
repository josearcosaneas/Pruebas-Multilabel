# Referencias: 
#	http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#example-svm-plot-svm-kernels-py
#	http://scikit-learn.org/stable/modules/svm.html#svm-kernels
#	http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#	http://scikit-learn.org/stable/modules/multiclass.html

'''
Funcion para unir los elementos de una lista en un solo elemento.
Dichos elementos seran string, que corresponderan al resumen de cada parrafo de 
una iniciativa.
El obejtivo de esta funcion es unir los diferentes resumenes de cada unos de los parrafos
para tener un solo string que represente el contenido de cada una de nuestras iniciativas
Entrada: lista de listas de strings
Salida: lista de np.string_
'''
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
'''
En las pruebas que estoy haciendo uso : 
	from sklearn.svm import LinearSVC ---->  OneVsRestClassifier(LinearSVC(random_state=0))	
Para usar otro nucleo seria de la siguente forma: 
	from sklearn.svm import SVC ---> classif = OneVsRestClassifier(SVC(kernel='linear'))
Los distintos kernels que vamos a usar son el kernel linear, rbf y polinomial.
	- linear
	- poly
	- rbf
'''
def clasificador(X_train, y_train, X_test, target_names): 
     
    lb = preprocessing.LabelBinarizer() 
     
    Y = lb.fit_transform(y_train) 
    # si no especificamos el kernel lo toma como 'rbf'  
    classifier = Pipeline([ 
        ('vectorizer',CountVectorizer()), 
        ('tfidf',TfidfTransformer()), 
        ('clf',OneVsRestClassifier(SVC(kernel='linear'))]) 
       
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

def clasificadorLinearSVC(X_train, y_train, X_test, target_names): 
     
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
