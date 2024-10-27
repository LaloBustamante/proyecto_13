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
import pandas as pd
import xgboost as xgb

from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from timeit import default_timer as timer


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


# Preprocesamiento y codificación para modelos
categorical_features = X.select_dtypes(include=['category']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=0.05), categorical_features)], remainder='passthrough')


'''
Definir y entrenar modelos

Modelos a considerar:
Regresión lineal (prueba de cordura)
Árbol de decisión
Bosque aleatorio
LightGBM 
XGBoost
'''


# Configuración de modelos y ajustes de hiperparámetros básicos
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=50, max_depth=10, learning_rate=0.1, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=50, max_depth=10, learning_rate=0.1, random_state=42)
}

# Entrenar y evaluar cada modelo
results = {}
for model_name, model in models.items():
    print(f"\nEntrenando modelo: {model_name}")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    
    start_time = timer()  # Tiempo de inicio
    pipeline.fit(X_train, y_train)
    end_time = timer()  # Tiempo final

    # Predicciones y evaluación
    y_pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'RMSE': rmse, 'R2': r2, 'Train Time (s)': end_time - start_time}

    print(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}, Train Time: {end_time - start_time:.2f} segundos")


'''
Análisis del Modelo:

Objetivo del Análisis: El análisis del modelo busca evaluar y comparar la precisión y el rendimiento de los modelos de predicción de precio 
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


# Medición de tiempo de predicción para el mejor modelo con pipeline completo
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', best_model)])

# Medir el tiempo de predicción en el conjunto de prueba
start_time = timer()
y_pred_best = best_pipeline.predict(X_test)
end_time = timer()

# Imprimir tiempo de predicción y métricas de evaluación del mejor modelo
rmse_best = mean_squared_error(y_test, y_pred_best, squared=False)
r2_best = r2_score(y_test, y_pred_best)
print(f"Tiempo de predicción del mejor modelo ({best_model_name}): {end_time - start_time:.4f} segundos")
print(f"RMSE del mejor modelo: {rmse_best:.4f}")
print(f"R2 del mejor modelo: {r2_best:.4f}")


'''
Conclusiones del Análisis de Modelos y Recomendaciones  
  
Linear Regression:

RMSE: 2890.02  
𝑅<sup>2</sup>: 0.6054  
Train Time: 2.50 segundos  
Análisis: La regresión lineal tiene el peor rendimiento en términos de error (RMSE) y coeficiente de determinación (𝑅<sup>2</sup>), lo cual es esperado, ya que este modelo es lineal y no captura bien las relaciones complejas. Sin embargo, es una buena base para comparar otros modelos.  
  
Decision Tree:  

RMSE: 2095.26  
𝑅<sup>2</sup>: 0.7926  
Train Time: 2.21 segundos  
Análisis: El árbol de decisión mejora considerablemente el rendimiento con respecto a la regresión lineal, reduciendo el RMSE y aumentando 𝑅<sup>2</sup>, lo cual indica un mejor ajuste. Aunque es un modelo rápido y simple, su rendimiento es inferior al de los otros modelos más avanzados.  

Random Forest:  

RMSE: 1994.40  
𝑅<sup>2</sup>: 0.8121  
Train Time: 23.14 segundos  
Análisis: El bosque aleatorio muestra una mejora significativa en comparación con el árbol de decisión, y alcanza un buen equilibrio entre RMSE y 𝑅<sup>2</sup>. Sin embargo, el tiempo de entrenamiento es considerablemente mayor debido al ensamblado de múltiples árboles.  

LightGBM:  

RMSE: 1925.82  
𝑅<sup>2</sup>: 0.8248  
Train Time: 1.41 segundos  
Análisis: LightGBM tiene un rendimiento excelente, logrando un RMSE bajo y un 𝑅<sup>2</sup> más alto en un tiempo de entrenamiento muy corto. Este modelo es eficiente y adecuado para grandes conjuntos de datos, mostrando una combinación de precisión y velocidad.

XGBoost:  

RMSE: 1772.34  
𝑅<sup>2</sup>: 0.8516  
Train Time: 2.63 segundos  
Análisis: XGBoost es el mejor en términos de precisión, obteniendo el RMSE más bajo y el 𝑅<sup>2</sup> más alto entre todos los modelos. Aunque el tiempo de entrenamiento es mayor que LightGBM, el incremento en precisión lo convierte en la mejor elección para este conjunto de datos.
'''

'''
Modelo Seleccionado   
  
Conclusión General:  

Mejor Modelo: XGBoost, ya que obtiene el mejor RMSE y R<sup>2</sup>, lo que indica una precisión superior en las predicciones.  

Modelos Alternativos: LightGBM es una excelente alternativa si el tiempo de entrenamiento es un factor importante, ya que ofrece una precisión cercana a XGBoost en menos tiempo.  

Recomendación: Considerar el uso de XGBoost para este caso, ya que su mayor precisión beneficia la predicción del valor de los autos usados, aunque LightGBM podría usarse si se requiere un modelo con un balance óptimo entre precisión y velocidad.
'''

