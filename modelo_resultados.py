# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 03:49:45 2023

@author: DESARROLLO
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Cargar los datos desde el archivo CSV
datos = pd.read_csv('dataset_original.csv')

# Dividir los datos en conjuntos de entrenamiento y prueba
X = datos['Sentence']
y_category = datos['Category']
y_subcategory = datos['Subcategory']
y_action = datos['Action']
X_train, X_test, y_category_train, y_category_test, y_subcategory_train, y_subcategory_test, y_action_train, y_action_test = train_test_split(
    X, y_category, y_subcategory, y_action, test_size=0.2, random_state=42)

# Crear una matriz de características usando CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Construir y entrenar el modelo de clasificación para la categoría
category_classifier = MultinomialNB()
category_classifier.fit(X_train_vectorized, y_category_train)

# Construir y entrenar el modelo de clasificación para la subcategoría
subcategory_classifier = MultinomialNB()
subcategory_classifier.fit(X_train_vectorized, y_subcategory_train)

# Construir y entrenar el modelo de clasificación para la acción
action_classifier = RandomForestClassifier()
action_classifier.fit(X_train_vectorized, y_action_train)

# Evaluar el modelo en el conjunto de prueba
category_predictions = category_classifier.predict(X_test_vectorized)
subcategory_predictions = subcategory_classifier.predict(X_test_vectorized)
action_predictions = action_classifier.predict(X_test_vectorized)

# Calcular las métricas para la categoría
category_accuracy = accuracy_score(y_category_test, category_predictions)
category_precision = precision_score(y_category_test, category_predictions, average='weighted')
category_recall = recall_score(y_category_test, category_predictions, average='weighted')
category_f1_score = f1_score(y_category_test, category_predictions, average='weighted')

# Calcular las métricas para la subcategoría
subcategory_accuracy = accuracy_score(y_subcategory_test, subcategory_predictions)
subcategory_precision = precision_score(y_subcategory_test, subcategory_predictions, average='weighted')
subcategory_recall = recall_score(y_subcategory_test, subcategory_predictions, average='weighted')
subcategory_f1_score = f1_score(y_subcategory_test, subcategory_predictions, average='weighted')

# Calcular las métricas para la acción
action_accuracy = accuracy_score(y_action_test, action_predictions)
action_precision = precision_score(y_action_test, action_predictions, average='weighted')
action_recall = recall_score(y_action_test, action_predictions, average='weighted')
action_f1_score = f1_score(y_action_test, action_predictions, average='weighted')

# Calcular la matriz de confusión para la categoría
category_confusion_matrix = confusion_matrix(y_category_test, category_predictions)

# Calcular la matriz de confusión para la subcategoría
subcategory_confusion_matrix = confusion_matrix(y_subcategory_test, subcategory_predictions)

# Calcular la matriz de confusión para la acción
action_confusion_matrix = confusion_matrix(y_action_test, action_predictions)

# Calcular la especificidad para la categoría
category_specificity = category_confusion_matrix[0, 0] / (category_confusion_matrix[0, 0] + category_confusion_matrix[0, 1])

# Calcular la especificidad para la subcategoría
subcategory_specificity = subcategory_confusion_matrix[0, 0] / (subcategory_confusion_matrix[0, 0] + subcategory_confusion_matrix[0, 1])

# Calcular la especificidad para la acción
action_specificity = action_confusion_matrix[0, 0] / (action_confusion_matrix[0, 0] + action_confusion_matrix[0, 1])

print("Métricas para la Categoría:")
print("Precisión: {:.2f}%".format(category_precision * 100))
print("Recall: {:.2f}%".format(category_recall * 100))
print("F1-score: {:.2f}%".format(category_f1_score * 100))
print("Exactitud (Accuracy): {:.2f}%".format(category_accuracy * 100))
print("Especificidad: {:.2f}%".format(category_specificity * 100))

print("\nMétricas para la Subcategoría:")
print("Precisión: {:.2f}%".format(subcategory_precision * 100))
print("Recall: {:.2f}%".format(subcategory_recall * 100))
print("F1-score: {:.2f}%".format(subcategory_f1_score * 100))
print("Exactitud (Accuracy): {:.2f}%".format(subcategory_accuracy * 100))
print("Especificidad: {:.2f}%".format(subcategory_specificity * 100))

print("\nMétricas para la Acción:")
print("Precisión: {:.2f}%".format(action_precision * 100))
print("Recall: {:.2f}%".format(action_recall * 100))
print("F1-score: {:.2f}%".format(action_f1_score * 100))
print("Exactitud (Accuracy): {:.2f}%".format(action_accuracy * 100))
print("Especificidad: {:.2f}%".format(action_specificity * 100))