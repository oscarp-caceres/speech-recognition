# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 04:56:53 2023

@author: DESARROLLO
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import speech_recognition as sr
from traductor_ingles import translate_to_english
# Crear un objeto reconocedor
r = sr.Recognizer()

# Función para transcribir el habla a texto
def transcribe_microphone():
    with sr.Microphone() as source:
        print("Di algo...")
        # Escuchar el audio del micrófono
        audio = r.listen(source)
        
        try:
            # Utilizar el reconocimiento de voz de Google para transcribir el habla
            text = r.recognize_google(audio, language='es-ES')
            return text
        except sr.UnknownValueError:
            print("No se pudo reconocer el habla")
        except sr.RequestError as e:
            print(f"Error al solicitar resultados del reconocimiento de voz de Google; {e}")

# Transcribir el habla en tiempo real
texto_transcrito = transcribe_microphone()

# Imprimir el texto transcribido
#print(texto_transcrito)

resultado = translate_to_english(texto_transcrito)
#print(resultado)

X_nuevos = [resultado]


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