import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# 1. Ampliar el conjunto de datos con más preguntas
data = [
    # Preguntas de Primero
    ("¿Cuánto es 2+2?", "Primero"),
    ("¿Cuánto es 5-3?", "Primero"),
    ("¿Cuál es el número siguiente a 3?", "Primero"),
    
    # Preguntas de Segundo
    ("¿Cuál es la capital de tu país?", "Segundo"),
    ("¿Quién fue Simón Bolívar?", "Segundo"),
    ("¿Qué día celebramos la independencia?", "Segundo"),
    
    # Preguntas de Tercero
    ("¿Qué es la fotosíntesis?", "Tercero"),
    ("Nombra una parte de la célula", "Tercero"),
    ("¿Qué son los mamíferos?", "Tercero"),
    
    # Preguntas de Cuarto
    ("Define una oración completa", "Cuarto"),
    ("¿Qué es un adjetivo?", "Cuarto"),
    ("¿Qué es un sustantivo?", "Cuarto"),
    
    # Preguntas de Quinto
    ("¿Cómo se conjuga el verbo 'to be' en inglés?", "Quinto"),
    ("¿Cómo se dice 'perro' en inglés?", "Quinto"),
    ("Escribe una oración en inglés con el verbo 'to be'", "Quinto")
]

# Separar preguntas y etiquetas de grado
preguntas, grados = zip(*data)

# 2. Vectorización de las preguntas
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preguntas)  # Convierte las preguntas a formato numérico

# 3. Convertir las etiquetas de texto a etiquetas numéricas
encoder = LabelEncoder()
y = encoder.fit_transform(grados)

# 4. Entrenar el modelo utilizando validación cruzada para mejor evaluación
modelo = MultinomialNB()

# Usar validación cruzada con 3 particiones (K-Fold Cross Validation)
scores = cross_val_score(modelo, X, y, cv=3)

# 5. Mostrar resultados de la validación cruzada
print("Precisión media de la validación cruzada:", np.mean(scores))

# 6. Entrenar el modelo en todos los datos para hacer predicciones finales
modelo.fit(X, y)

# Probar con una pregunta nueva
nueva_pregunta = ["¿Qué es un sustantivo?"]
nueva_pregunta_vec = vectorizer.transform(nueva_pregunta)
prediccion = modelo.predict(nueva_pregunta_vec)

# Convertir predicción numérica de vuelta a texto
prediccion_texto = encoder.inverse_transform(prediccion)

print("Predicción para la nueva pregunta:", prediccion_texto[0])
