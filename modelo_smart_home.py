# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:46:12 2023

@author: DESARROLLO
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
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

joblib.dump(vectorizer, 'vectorizer.joblib')


# Construir y entrenar el modelo de clasificación para la categoría
category_classifier = MultinomialNB()
category_classifier.fit(X_train_vectorized, y_category_train)


# Guardar el modelo de clasificación para la categoría
joblib.dump(category_classifier, 'category_model.joblib')

# Construir y entrenar el modelo de clasificación para la subcategoría
subcategory_classifier = MultinomialNB()
subcategory_classifier.fit(X_train_vectorized, y_subcategory_train)

# Guardar el modelo de clasificación para la subcategoría
joblib.dump(subcategory_classifier, 'subcategory_model.joblib')

# Construir y entrenar el modelo de clasificación para la acción
action_classifier = RandomForestClassifier()
action_classifier.fit(X_train_vectorized, y_action_train)

# Guardar el modelo de clasificación para la acción
joblib.dump(action_classifier, 'action_model.joblib')

# Evaluar el modelo en el conjunto de prueba
category_predictions = category_classifier.predict(X_test_vectorized)
subcategory_predictions = subcategory_classifier.predict(X_test_vectorized)
action_predictions = action_classifier.predict(X_test_vectorized)

# Generar la matriz de confusión sin normalizar
category_confusion_matrix = confusion_matrix(y_category_test, category_predictions)
subcategory_confusion_matrix = confusion_matrix(y_subcategory_test, subcategory_predictions)
action_confusion_matrix = confusion_matrix(y_action_test, action_predictions)

# Generar la matriz de confusión normalizada
category_confusion_matrix_normalized = category_confusion_matrix.astype('float') / category_confusion_matrix.sum(axis=1)[:, np.newaxis]
subcategory_confusion_matrix_normalized = subcategory_confusion_matrix.astype('float') / subcategory_confusion_matrix.sum(axis=1)[:, np.newaxis]
action_confusion_matrix_normalized = action_confusion_matrix.astype('float') / action_confusion_matrix.sum(axis=1)[:, np.newaxis]

# Evaluar el modelo en el conjunto de prueba
category_predictions = category_classifier.predict(X_test_vectorized)
subcategory_predictions = subcategory_classifier.predict(X_test_vectorized)
action_predictions = action_classifier.predict(X_test_vectorized)

# Calcular la precisión de los modelos
category_accuracy = accuracy_score(y_category_test, category_predictions)
subcategory_accuracy = accuracy_score(y_subcategory_test, subcategory_predictions)
action_accuracy = accuracy_score(y_action_test, action_predictions)

print("Precisión de la Categoría: {:.2f}%".format(category_accuracy * 100))
print("Precisión de la Subcategoría: {:.2f}%".format(subcategory_accuracy * 100))
print("Precisión de la Acción: {:.2f}%".format(action_accuracy * 100))

# Generar la matriz de confusión en forma de tabla
def confusion_matrix_to_table(confusion_matrix, labels):
    df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    return df

category_labels = category_classifier.classes_
subcategory_labels = subcategory_classifier.classes_
action_labels = action_classifier.classes_

category_confusion_table = confusion_matrix_to_table(category_confusion_matrix, category_labels)
subcategory_confusion_table = confusion_matrix_to_table(subcategory_confusion_matrix, subcategory_labels)
action_confusion_table = confusion_matrix_to_table(action_confusion_matrix, action_labels)

print("Matriz de confusión de Categoría (sin normalizar):\n", category_confusion_table)
print("Matriz de confusión de Subcategoría (sin normalizar):\n", subcategory_confusion_table)
print("Matriz de confusión de Acción (sin normalizar):\n", action_confusion_table)

#
# Generar la matriz de confusión sin normalizar
category_confusion_matrix = confusion_matrix(y_category_test, category_predictions)
subcategory_confusion_matrix = confusion_matrix(y_subcategory_test, subcategory_predictions)
action_confusion_matrix = confusion_matrix(y_action_test, action_predictions)

# Función para mostrar la matriz de confusión como imagen
def show_confusion_matrix(confusion_matrix, labels, title):
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicate Label')
    plt.ylabel('True Label')
    plt.show()

# Mostrar la matriz de confusión sin normalizar como imagen
category_labels = category_classifier.classes_
subcategory_labels = subcategory_classifier.classes_
action_labels = action_classifier.classes_

show_confusion_matrix(category_confusion_matrix, category_labels, 'Confusion matrix - Category')
show_confusion_matrix(subcategory_confusion_matrix, subcategory_labels, 'Confusion Matrix - Subcategory')
show_confusion_matrix(action_confusion_matrix, action_labels, 'Confusion matrix - Action')

# Generar la matriz de confusión normalizada en forma de imagen
def plot_confusion_matrix(confusion_matrix, labels, title):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Prediction")
    plt.ylabel("True Label")
    plt.show()

plot_confusion_matrix(category_confusion_matrix_normalized, category_labels, "Normalized confusion matrix - Category")
plot_confusion_matrix(subcategory_confusion_matrix_normalized, subcategory_labels, "Normalized confusion matrix - Subcategory")
plot_confusion_matrix(action_confusion_matrix_normalized, action_labels, "Normalized confusion matrix - Action")

#*Make predictions on new data
nuevas_oraciones = ["Can you turn on the heating in the library in an hour?"]
nuevas_oraciones_vectorized = vectorizer.transform(nuevas_oraciones)
category_predictions = category_classifier.predict(nuevas_oraciones_vectorized)
subcategory_predictions = subcategory_classifier.predict(nuevas_oraciones_vectorized)
action_predictions = action_classifier.predict(nuevas_oraciones_vectorized)



#*Print predictions
print("Category Predictions:", category_predictions)
print("Subcategory Predictions:", subcategory_predictions)
print("Action Predictions:", action_predictions)
