# IA para Reconocimiento de Grado Escolar

Este proyecto implementa una **Inteligencia Artificial (IA)** en Python que puede determinar el grado escolar de un estudiante basado en una serie de preguntas que abarcan cinco materias: Matemáticas, Sociales, Naturales, Español e Inglés. El modelo utiliza el algoritmo de clasificación **Naive Bayes Multinomial** de `sklearn` y realiza predicciones a partir de preguntas predefinidas para evaluar los grados de **Primero** a **Quinto** de primaria.

## Características
- **Cinco materias**: Matemáticas, Sociales, Naturales, Español e Inglés.
- **Predicción de grado escolar**: El modelo determina si un estudiante pertenece a Primero, Segundo, Tercero, Cuarto o Quinto grado.
- **Validación cruzada**: Se utiliza validación cruzada para evaluar el rendimiento del modelo.
- **Algoritmo utilizado**: Naive Bayes Multinomial, ideal para problemas de clasificación de texto.

## Requisitos

Para ejecutar este proyecto necesitas tener instalados los siguientes paquetes:

- Python 3.x
- scikit-learn
- numpy

Puedes instalar las dependencias con el siguiente comando:

```bash
pip install scikit-learn numpy  
```
## Estructura del Proyecto
- **Entrenamiento del modelo**: El modelo se entrena con un conjunto de preguntas predefinidas relacionadas con cada materia y grado escolar.
- **Validación cruzada**: Se utiliza la validación cruzada con 3 particiones para evaluar la precisión del modelo.
- **Predicción**: El modelo predice el grado escolar correspondiente a una nueva pregunta proporcionada.
## Cómo Ejecutar el Proyecto
1. Clona el repositorio:
```bash
git clone https://github.com/tu_usuario/Prototype.git
```
2. Accede al directorio del Proyecto:
```
cd Prototype
```
3. Ejecuta el script principal:
```
python main.py
```
## Ejemplo de Uso
El script incluye un conjunto de datos con preguntas predefinidas para los cinco grados de primaria. Puedes agregar tus propias preguntas o modificar las existentes.

## Entrenamiento y Validación
El modelo se entrena usando las preguntas y luego se evalúa mediante validación cruzada:

```python
# El conjunto de datos con preguntas
data = [
    ("¿Cuánto es 2+2?", "Primero"),
    ("¿Cuál es la capital de tu país?", "Segundo"),
    ("¿Qué es la fotosíntesis?", "Tercero"),
    ("Define una oración completa", "Cuarto"),
    ("¿Cómo se conjuga el verbo 'to be' en inglés?", "Quinto")
]

# Vectorización de las preguntas y codificación de etiquetas
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preguntas)
encoder = LabelEncoder()
y = encoder.fit_transform(grados)

# Entrenamiento y validación cruzada
modelo = MultinomialNB()
scores = cross_val_score(modelo, X, y, cv=3)
print("Precisión media:", np.mean(scores))
```

## Predicción de una nueva pregunta
Puedes probar el modelo con una nueva pregunta:

```python
nueva_pregunta = ["¿Qué es un sustantivo?"]
nueva_pregunta_vec = vectorizer.transform(nueva_pregunta)
prediccion = modelo.predict(nueva_pregunta_vec)
prediccion_texto = encoder.inverse_transform(prediccion)
print("Predicción para la nueva pregunta:", prediccion_texto[0])
```
## Authors

- [@Mailet2997](https://github.com/mailet2997)
- [@ThomBertel](https://www.instagram.com/thomcruzbertel)


