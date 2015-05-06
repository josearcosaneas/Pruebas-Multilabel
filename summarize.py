"""
@author Jose Arcos Aneas

Prueba de uso de la libreia summa de python para resumir textos.
Para la prueba uso un archivo XML.

"""
# -*- coding: utf8 -*-

from summa import summarizer


from xml.dom.minidom import parse
import numpy as np

#Lectura de todos los tags 'parrafo' de un archivo XML
midom=parse("/home/blunt/Escritorio/TFG-SIDP/iniciativas/DSCA080077_3.xml")
elements = midom.getElementsByTagName('parrafo')
resultList = []
if len(elements) != 0:
    for i in range(0,len(elements)):
        resultList.extend([elements[i].childNodes[0].nodeValue])

# los introducimos en una lista y los hacemos string
sentence =[]

for i in resultList:
    sentence.append(i)
sentence=str(sentence)

# Mostramos el texto y el resumen, usando el lenguaje Español para especificar las keywords

print sentence
print "********"*23
print """   Resumen a partir de aquí """
print "********"*23
print summarizer.summarize(sentence,language='spanish' )
