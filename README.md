# IA para Reconocimiento de Grado Escolar

Este proyecto implementa una **Inteligencia Artificial (IA)** en Python que puede determinar el grado escolar de un estudiante basado en una serie de preguntas que abarcan cinco materias: Matemáticas, Sociales, Naturales, Español e Inglés. El modelo utiliza el algoritmo de clasificación **Naive Bayes Multinomial** de `sklearn` y realiza predicciones a partir de preguntas predefinidas para evaluar los grados de **Primero** a **Quinto** de primaria.

## Características
- **Cinco materias**: Matemáticas, Sociales, Naturales, Español e Inglés.
- **Predicción de grado escolar**: El modelo determina si un estudiante pertenece a Primero, Segundo, Tercero, Cuarto o Quinto grado.
- **Validación cruzada**: Se utiliza validación cruzada para evaluar el rendimiento del modelo.
- **Algoritmo utilizado**: Naive Bayes Multinomial, ideal para problemas de clasificación de texto.

### Cambios en la última actualización:
- Agregado un sistema interactivo para que el usuario responda preguntas de selección única en la consola.
- Ampliado el conjunto de datos con nuevas preguntas y opciones para todos los grados.
- Añadido preprocesamiento de respuestas del usuario para integrarlas con el modelo de predicción.
- Mejorada la robustez del sistema mediante validación de entradas en tiempo real.
- Refinado el flujo del programa para presentar resultados de manera más clara.


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
- **Interactividad**: El programa permite al usuario responder preguntas de selección única para obtener una predicción del grado escolar.
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
python Prototype.py
```
## Estructura del Proyecto
**Nuevas funcionalidades**
- **Sistema interactivo para el usuario**: Ahora puedes responder preguntas de selección única mediante opciones numéricas (1-4). El sistema validará las respuestas para asegurarse de que sean válidas.
- **Ampliación del conjunto de datos**: Se duplicó el número de preguntas y respuestas para abarcar un mayor rango de temas en cada grado escolar.
- **Preprocesamiento de nuevas respuestas**: Las respuestas seleccionadas por el usuario se procesan y vectorizan antes de ser evaluadas por el modelo.
- **Integración con el modelo Naive Bayes**: El modelo ahora utiliza las respuestas seleccionadas por el usuario para predecir el grado escolar de manera interactiva.
- **Mejoras en la presentación de resultados**: Se mejoró la presentación de los resultados para una experiencia de usuario más clara y comprensible.

## Ejemplo de Salida

```Bienvenido al sistema de predicción de grado escolar.
Por favor, responde las siguientes preguntas seleccionando una opción numérica:

Pregunta 1: ¿Cuánto es 2+2?
1. 1
2. 2
3. 4
4. 5
Selecciona una opción (1-4): 3

Pregunta 2: ¿Qué forma tiene una pelota?
1. Cuadrada
2. Redonda
3. Triangular
4. Rectangular
Selecciona una opción (1-4): 2

...

El modelo predice que el grado escolar es: Primero
```


## Authors

- [@Mailet2997](https://github.com/mailet2997)
- [@ThomBertel](https://www.instagram.com/thomcruzbertel)
- [@AudithGuerrero](https://www.instagram.com/audithmariaguerrero)


