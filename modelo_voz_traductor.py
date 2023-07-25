# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 04:49:35 2023

@author: DESARROLLO
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from transformar_voz_texto import transcribe_microphone 
from traductor_ingles import translate_to_english

# Cargar los datos desde un archivo CSV o cualquier otra fuente
nuevos_datos = pd.read_csv('dataset_prueba.csv')

#Transformar el habla a texto
texto_transcrito = transcribe_microphone()

#Traducir el texto de ES a EN

resultado = translate_to_english(texto_transcrito)
print(resultado)
# Obtener las características de los nuevos datos
X_nuevos = ["Illuminate the kitchen today."]


# Cargar el vectorizador y los modelos guardados
vectorizer = joblib.load('vectorizer.joblib')
loaded_category_model = joblib.load('category_model.joblib')
loaded_subcategory_model = joblib.load('subcategory_model.joblib')
loaded_action_model = joblib.load('action_model.joblib')

# Vectorizar los nuevos datos
X_nuevos_vectorized = vectorizer.transform(X_nuevos)

# Hacer predicciones en los nuevos datos
category_predictions = loaded_category_model.predict(X_nuevos_vectorized)
subcategory_predictions = loaded_subcategory_model.predict(X_nuevos_vectorized)
action_predictions = loaded_action_model.predict(X_nuevos_vectorized)

# Mostrar las predicciones
results = pd.DataFrame({'Sentence': X_nuevos, 'Category': category_predictions, 'Subcategory': subcategory_predictions, 'Action': action_predictions})
print(results)