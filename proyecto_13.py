'''
Descripción del proyecto
Rusty Bargain es un servicio de venta de coches de segunda mano que está desarrollando una app para atraer a nuevos clientes. Gracias a esa app, puedes averiguar rápidamente el valor de mercado de tu coche. Tienes acceso al historial, especificaciones técnicas, versiones de equipamiento y precios. Tienes que crear un modelo que determine el valor de mercado.

A Rusty Bargain le interesa:

la calidad de la predicción
la velocidad de la predicción
el tiempo requerido para el entrenamiento

Instrucciones del proyecto
Descarga y examina los datos.
Entrena diferentes modelos con varios hiperparámetros (debes hacer al menos dos modelos diferentes, pero más es mejor. Recuerda, varias implementaciones de potenciación del gradiente no cuentan como modelos diferentes). El punto principal de este paso es comparar métodos de potenciación del gradiente con bosque aleatorio, árbol de decisión y regresión lineal.
Analiza la velocidad y la calidad de los modelos.
Observaciones:

Utiliza la métrica RECM para evaluar los modelos.
La regresión lineal no es muy buena para el ajuste de hiperparámetros, pero es perfecta para hacer una prueba de cordura de otros métodos. Si la potenciación del gradiente funciona peor que la regresión lineal, definitivamente algo salió mal.
Aprende por tu propia cuenta sobre la librería LightGBM y sus herramientas para crear modelos de potenciación del gradiente (gradient boosting).
Idealmente, tu proyecto debe tener regresión lineal para una prueba de cordura, un algoritmo basado en árbol con ajuste de hiperparámetros (preferiblemente, bosque aleatorio), LightGBM con ajuste de hiperparámetros (prueba un par de conjuntos), y CatBoost y XGBoost con ajuste de hiperparámetros (opcional).
Toma nota de la codificación de características categóricas para algoritmos simples. LightGBM y CatBoost tienen su implementación, pero XGBoost requiere OHE.
Puedes usar un comando especial para encontrar el tiempo de ejecución del código de celda en Jupyter Notebook. Encuentra ese comando.
Dado que el entrenamiento de un modelo de potenciación del gradiente puede llevar mucho tiempo, cambia solo algunos parámetros del modelo.
Si Jupyter Notebook deja de funcionar, elimina las variables excesivas por medio del operador del:

  del features_train
  
Descripción de los datos
El dataset está almacenado en el archivo /datasets/car_data.csv. descargar dataset.

Características

DateCrawled — fecha en la que se descargó el perfil de la base de datos
VehicleType — tipo de carrocería del vehículo
RegistrationYear — año de matriculación del vehículo
Gearbox — tipo de caja de cambios
Power — potencia (CV)
Model — modelo del vehículo
Mileage — kilometraje (medido en km de acuerdo con las especificidades regionales del conjunto de datos)
RegistrationMonth — mes de matriculación del vehículo
FuelType — tipo de combustible
Brand — marca del vehículo
NotRepaired — vehículo con o sin reparación
DateCreated — fecha de creación del perfil
NumberOfPictures — número de fotos del vehículo
PostalCode — código postal del propietario del perfil (usuario)
LastSeen — fecha de la última vez que el usuario estuvo activo
Objetivo

Price — precio (en euros)

Evaluación del proyecto
Hemos definido los criterios de evaluación para el proyecto. Léelos con atención antes de pasar al ejercicio.

Esto es en lo que se fijarán los revisores al examinar tu proyecto:

¿Seguiste todos los pasos de las instrucciones?
¿Cómo preparaste los datos?
¿Qué modelos e hiperparámetros consideraste?
¿Conseguiste evitar la duplicación del código?
¿Cuáles son tus hallazgos?
¿Mantuviste la estructura del proyecto?
¿Mantuviste el código ordenado?
  
'''


'''
1. Preparación de Datos
'''


# Importar librerías
import lightgbm as lgb
import numpy as np
import timeit as t
import pandas as pd

from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder



# Cargar los datos
df = pd.read_csv('datasets/car_data.csv')


# Exploración inicial
print(df.info())
display(df.head())


'''
Revisión y limpieza:

Eliminar columnas que probablemente no serán útiles para el modelo, como DateCrawled, DateCreated, LastSeen, PostalCode, y NumberOfPictures.

Manejar valores nulos en columnas relevantes como VehicleType, Gearbox, FuelType, y NotRepaired.

Convertir valores categóricos a tipo category para optimizar la memoria.

Analizar y tratar posibles errores en RegistrationYear y Power.
'''


# Eliminar columnas innecesarias
df.drop(['DateCrawled', 'DateCreated', 'LastSeen', 'PostalCode', 'NumberOfPictures'], axis=1, inplace=True)


# Rellenar valores faltantes en columnas categóricas con un valor específico o "Unknown"
df['VehicleType'].fillna('Unknown', inplace=True)
df['Gearbox'].fillna('Unknown', inplace=True)
df['FuelType'].fillna('Unknown', inplace=True)
df['NotRepaired'].fillna('Unknown', inplace=True)


# Tratar errores en 'RegistrationYear' y 'Power'
df = df[(df['RegistrationYear'] >= 1886) & (df['RegistrationYear'] <= 2024)]
df = df[df['Power'].between(1, 1000)]


# Convertir columnas categóricas
for col in ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'NotRepaired']:
    df[col] = df[col].astype('category')


# Confirmar limpieza
print(df.info())
display(df.describe())


'''
2. Entrenamiento del Modelo
Objetivo: Probar y ajustar varios modelos, incluyendo regresión lineal (para referencia), árboles de decisión, bosque aleatorio, y 
LightGBM.
'''


#  Dividir los datos en conjuntos de entrenamiento y prueba
# Definir características y objetivo
X = df.drop('Price', axis=1)
y = df['Price']


# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Preprocesamiento de características categóricas
# Codificación de características categóricas para modelos que lo requieren
categorical_features = X.select_dtypes(include=['category']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough')


'''
Definir y entrenar modelos

Modelos a considerar:
Regresión lineal (prueba de cordura)
Árbol de decisión
Bosque aleatorio
LightGBM (recomendado para datos de alta dimensionalidad y muchas categorías)
'''


# Lista de modelos a probar
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1)
}

# Entrenar y evaluar cada modelo
results = {}
for model_name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    # Entrenar
    pipeline.fit(X_train, y_train)

    # Predicciones
    y_pred = pipeline.predict(X_test)

    # Evaluación
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'RMSE': rmse, 'R2': r2}

    print(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")


'''
Análisis del Modelo
Objetivo del Análisis El análisis del modelo busca evaluar y comparar la precisión y el rendimiento de los modelos de predicción de precio 
de vehículos de segunda mano, con énfasis en la métrica RMSE (raíz cuadrada del error cuadrático medio) y el coeficiente de determinación 
𝑅2. Estos resultados ayudarán a determinar el modelo que mejor equilibra precisión y eficiencia para ser implementado en la app de 
Rusty Bargain.
'''

# Resumen de resultados
results_df = pd.DataFrame(results).T
display(results_df.sort_values(by='RMSE'))


# Medición de tiempo de predicción para el mejor modelo
best_model_name = results_df['RMSE'].idxmin()
best_model = models[best_model_name]


# Tiempo de predicción
t.timeit(best_model.predict(X_test))


'''
Conclusiones del Análisis de Modelos y Recomendaciones

A partir de los resultados obtenidos para los modelos de regresión lineal y árbol de decisión:

Regresión Lineal

RMSE: 2914.06
R²: 0.5988
Este modelo explica aproximadamente el 59.88% de la varianza en los precios y presenta un error moderado, pero no es el modelo más preciso. 
Debido a su simplicidad, podría ser útil en aplicaciones donde se requiera rapidez en la predicción, aunque en este caso no se aprovechan 
las relaciones no lineales entre variables, resultando en una predicción menos precisa comparada con otros métodos.

Árbol de Decisión

RMSE: 2146.58
R²: 0.7823
El árbol de decisión es significativamente más preciso que la regresión lineal. Con un 𝑅2 de 78.23%, este modelo es capaz de capturar 
mejor las variaciones en el precio, lo cual lo hace más adecuado para nuestro caso. Esto sugiere que este modelo capta relaciones 
complejas en los datos que pueden pasar desapercibidas en una regresión lineal.
'''

'''
Modelo Seleccionado
Modelo Final: Árbol de Decisión

Se seleccionó el árbol de decisión como modelo final debido a los siguientes factores:

Precisión: 
Este modelo superó a la regresión lineal en términos de 𝑅2 y RMSE, mostrando una mayor capacidad para predecir precios con precisión. 

Capacidad de Captura de Relaciones Complejas: 
El árbol de decisión es más adecuado para datos con relaciones no lineales, permitiéndole captar patrones relevantes en los datos de Rusty 
Bargain.

Balance de Velocidad y Exactitud: Aunque es más complejo que la regresión lineal, el árbol de decisión sigue siendo relativamente rápido 
en comparación con otros modelos avanzados, lo cual es beneficioso en un contexto donde la velocidad de predicción también es relevante.
'''

