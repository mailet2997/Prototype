import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

# Conjunto de datos ampliado con preguntas y opciones
data = [
    ("¿Cuánto es 2+2?", "Primero", ["1", "2", "4", "5"]),
    ("¿Qué forma tiene una pelota?", "Primero", ["Cuadrada", "Redonda", "Triangular", "Rectangular"]),
    ("¿De qué color es el cielo durante el día?", "Primero", ["Rojo", "Azul", "Negro", "Verde"]),
    ("¿Cómo se llama el día después de hoy?", "Primero", ["Ayer", "Mañana", "Hoy", "Nunca"]),
    ("¿Qué viene después del número 5?", "Primero", ["4", "6", "8", "2"]),

    ("¿Cuál es la capital de Colombia?", "Segundo", ["Bogotá", "Madrid", "Ciudad de México", "Lima"]),
    ("¿Qué figura tiene cuatro lados iguales?", "Segundo", ["Triángulo", "Círculo", "Cuadrado", "Pentágono"]),
    ("¿Qué planeta habitamos?", "Segundo", ["Marte", "Venus", "Tierra", "Saturno"]),
    ("¿Cómo se llama el hermano de tu mamá?", "Segundo", ["Abuelo", "Tío", "Padre", "Primo"]),
    ("¿Qué animal tiene trompa y es muy grande?", "Segundo", ["Tigre", "Elefante", "Zorro", "Cebra"]),

    ("¿Qué es la fotosíntesis?", "Tercero", ["Respiración de animales", "Proceso en plantas", "Erosión de suelo", "Ciclo del agua"]),
    ("¿Cuántos continentes hay en el mundo?", "Tercero", ["5", "6", "7", "8"]),
    ("¿Qué país tiene forma de bota?", "Tercero", ["Brasil", "Italia", "Egipto", "Japón"]),
    ("¿Qué gas respiran las plantas para hacer fotosíntesis?", "Tercero", ["Oxígeno", "Dióxido de carbono", "Nitrógeno", "Hidrógeno"]),
    ("¿Cuál es la moneda de Colombia?", "Tercero", ["Peso", "Euro", "Dólar", "Yen"]),

    ("Define una oración completa", "Cuarto", ["Conjunto de palabras con sentido", "Una palabra suelta", "Un dibujo", "Un número"]),
    ("¿Cuál es el río más largo del mundo?", "Cuarto", ["Amazonas", "Nilo", "Yangtsé", "Danubio"]),
    ("¿Qué es una fracción?", "Cuarto", ["Parte de un todo", "Un número entero", "Un animal", "Una máquina"]),
    ("¿Qué órgano se encarga de bombear la sangre?", "Cuarto", ["Pulmones", "Hígado", "Corazón", "Cerebro"]),
    ("¿Qué energía proviene del sol?", "Cuarto", ["Solar", "Eólica", "Hidráulica", "Térmica"]),

    ("¿Cómo se conjuga el verbo 'to be' en inglés?", "Quinto", ["is, are, was", "do, does, did", "am, is, are", "will, shall, should"]),
    ("¿Cuál es el sistema nervioso central del cuerpo?", "Quinto", ["Cerebro", "Estómago", "Huesos", "Pulmones"]),
    ("¿Quién escribió 'El Quijote'?", "Quinto", ["Cervantes", "Shakespeare", "Borges", "Neruda"]),
    ("¿Cuál es la fórmula química del agua?", "Quinto", ["H2O", "CO2", "O2", "NaCl"]),
    ("¿Cuál es la raíz cuadrada de 144?", "Quinto", ["12", "14", "16", "18"])
]

# Separar preguntas, grados y opciones
preguntas, grados, opciones = zip(*data)

# Vectorizar preguntas y codificar grados
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preguntas)
encoder = LabelEncoder()
y = encoder.fit_transform(grados)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# Validación cruzada
scores = cross_val_score(modelo, X, y, cv=4)
print(f"Precisión media del modelo: {np.mean(scores):.2f}")

# Sistema interactivo para responder preguntas
print("\nBienvenido al sistema de predicción de grado escolar.")
print("Por favor, responde las siguientes preguntas seleccionando una opción numérica:")

nueva_pregunta = []
for i, pregunta in enumerate(preguntas):
    print(f"\nPregunta {i + 1}: {pregunta}")
    for j, opcion in enumerate(opciones[i]):
        print(f"{j + 1}. {opcion}")
    
    # Validar entrada del usuario
    while True:
        try:
            respuesta = int(input("Selecciona una opción (1-4): "))
            if 1 <= respuesta <= len(opciones[i]):
                nueva_pregunta.append(opciones[i][respuesta - 1])
                break
            else:
                print("Por favor selecciona una opción válida.")
        except ValueError:
            print("Entrada inválida. Por favor ingresa un número.")

# Convertir la nueva pregunta a un formato entendible por el modelo
nueva_pregunta_str = " ".join(nueva_pregunta)
nueva_pregunta_vec = vectorizer.transform([nueva_pregunta_str])

# Predecir el grado escolar
prediccion = modelo.predict(nueva_pregunta_vec)
grado_predicho = encoder.inverse_transform(prediccion)
print(f"\nEl modelo predice que el grado escolar es: {grado_predicho[0]}")
