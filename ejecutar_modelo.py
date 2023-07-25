# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:54:45 2023

@author: DESARROLLO
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Cargar los datos desde un archivo CSV o cualquier otra fuente
nuevos_datos = pd.read_csv('dataset_prueba.csv')

# Obtener las características de los nuevos datos
X_nuevos = nuevos_datos['Sentence']

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