# flake8: noqa: E501
"""
Sistema de predicción de incumplimiento de pago de clientes.
Modelo de clasificación basado en regresión logística con optimización de hiperparámetros.
"""

import gzip
import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def cargar_datos_crudos(archivo_comprimido: str) -> pd.DataFrame:
    """
    Carga y preprocesa los datos desde archivo ZIP.
    
    Args:
        archivo_comprimido: Ruta al archivo CSV comprimido
        
    Returns:
        DataFrame limpio y procesado
    """
    datos = pd.read_csv(archivo_comprimido, compression="zip").copy()

    # Renombrar columna objetivo
    datos.rename(columns={"default payment next month": "default"}, inplace=True)
    
    # Eliminar columna ID si existe
    if "ID" in datos.columns:
        datos.drop(columns=["ID"], inplace=True)

    # Filtrar registros con valores N/A en EDUCATION y MARRIAGE
    datos = datos[(datos["EDUCATION"] != 0) & (datos["MARRIAGE"] != 0)]
    
    # Consolidar niveles superiores de educación
    datos["EDUCATION"] = datos["EDUCATION"].apply(lambda valor: 4 if valor > 4 else valor)

    return datos.dropna()


def separar_caracteristicas_objetivo(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa las características de la variable objetivo.
    
    Args:
        dataset: DataFrame completo
        
    Returns:
        Tupla con (características, objetivo)
    """
    caracteristicas = dataset.drop(columns=["default"])
    objetivo = dataset["default"]
    return caracteristicas, objetivo


def construir_optimizador(numero_caracteristicas: int) -> GridSearchCV:
    """
    Construye el pipeline de ML y el optimizador de hiperparámetros.
    
    Args:
        numero_caracteristicas: Número total de características
        
    Returns:
        Objeto GridSearchCV configurado
    """
    # Definir columnas categóricas
    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    
    # Construir transformador de columnas
    transformador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("num", MinMaxScaler(), [])
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    # Modelo de clasificación
    clasificador = LogisticRegression(max_iter=1000, random_state=42)

    # Pipeline completo
    flujo_trabajo = Pipeline(
        steps=[
            ("prep", transformador),
            ("kbest", SelectKBest(score_func=f_regression)),
            ("clf", clasificador),
        ]
    )

    # Espacio de búsqueda de hiperparámetros
    espacio_parametros = {
        "kbest__k": list(range(1, numero_caracteristicas + 1)),
        "clf__C": [0.1, 1, 10],
        "clf__solver": ["liblinear", "lbfgs"],
    }

    # Optimizador con validación cruzada
    optimizador = GridSearchCV(
        estimator=flujo_trabajo,
        param_grid=espacio_parametros,
        scoring="balanced_accuracy",
        cv=10,
        refit=True,
        n_jobs=-1,
    )
    
    return optimizador


def calcular_metricas_rendimiento(etiqueta_conjunto: str, valores_reales, valores_predichos) -> dict:
    """
    Calcula métricas de rendimiento del modelo.
    
    Args:
        etiqueta_conjunto: 'train' o 'test'
        valores_reales: Valores reales del objetivo
        valores_predichos: Valores predichos por el modelo
        
    Returns:
        Diccionario con las métricas
    """
    metricas = {
        "type": "metrics",
        "dataset": etiqueta_conjunto,
        "precision": precision_score(valores_reales, valores_predichos, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(valores_reales, valores_predichos),
        "recall": recall_score(valores_reales, valores_predichos, zero_division=0),
        "f1_score": f1_score(valores_reales, valores_predichos, zero_division=0),
    }
    return metricas


def generar_matriz_confusion(etiqueta_conjunto: str, valores_reales, valores_predichos) -> dict:
    """
    Genera la matriz de confusión en formato de diccionario.
    
    Args:
        etiqueta_conjunto: 'train' o 'test'
        valores_reales: Valores reales del objetivo
        valores_predichos: Valores predichos por el modelo
        
    Returns:
        Diccionario con la matriz de confusión
    """
    matriz = confusion_matrix(valores_reales, valores_predichos)
    
    resultado = {
        "type": "cm_matrix",
        "dataset": etiqueta_conjunto,
        "true_0": {
            "predicted_0": int(matriz[0, 0]), 
            "predicted_1": int(matriz[0, 1])
        },
        "true_1": {
            "predicted_0": int(matriz[1, 0]), 
            "predicted_1": int(matriz[1, 1])
        },
    }
    return resultado


def persistir_modelo(modelo_entrenado, ruta_salida: str) -> None:
    """
    Guarda el modelo entrenado en formato comprimido.
    
    Args:
        modelo_entrenado: Modelo entrenado
        ruta_salida: Ruta donde guardar el modelo
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with gzip.open(ruta_salida, "wb") as archivo:
        pickle.dump(modelo_entrenado, archivo)


def guardar_resultados(lista_resultados: list, ruta_archivo: str) -> None:
    """
    Guarda los resultados en formato JSON línea por línea.
    
    Args:
        lista_resultados: Lista de diccionarios con resultados
        ruta_archivo: Ruta del archivo de salida
    """
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        for resultado in lista_resultados:
            archivo.write(json.dumps(resultado) + "\n")


def ejecutar_pipeline_completo() -> None:
    """
    Ejecuta el pipeline completo de entrenamiento y evaluación.
    """
    # Cargar y limpiar datos
    datos_entrenamiento = cargar_datos_crudos("files/input/train_data.csv.zip")
    datos_prueba = cargar_datos_crudos("files/input/test_data.csv.zip")

    # Separar características y objetivo
    X_entrenamiento, y_entrenamiento = separar_caracteristicas_objetivo(datos_entrenamiento)
    X_prueba, y_prueba = separar_caracteristicas_objetivo(datos_prueba)

    # Construir optimizador
    buscador_parametros = construir_optimizador(numero_caracteristicas=X_entrenamiento.shape[1])

    # Identificar columnas numéricas y categóricas
    cols_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    cols_numericas = [col for col in X_entrenamiento.columns if col not in cols_categoricas]

    # Actualizar el preprocesador con las columnas correctas
    buscador_parametros.estimator.named_steps["prep"] = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cols_categoricas),
            ("num", MinMaxScaler(), cols_numericas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Entrenar modelo con búsqueda de hiperparámetros
    buscador_parametros.fit(X_entrenamiento, y_entrenamiento)

    # Guardar modelo entrenado
    persistir_modelo(buscador_parametros, "files/models/model.pkl.gz")

    # Realizar predicciones
    predicciones_entrenamiento = buscador_parametros.predict(X_entrenamiento)
    predicciones_prueba = buscador_parametros.predict(X_prueba)

    # Calcular métricas y matrices de confusión
    resultados_evaluacion = [
        calcular_metricas_rendimiento("train", y_entrenamiento, predicciones_entrenamiento),
        calcular_metricas_rendimiento("test", y_prueba, predicciones_prueba),
        generar_matriz_confusion("train", y_entrenamiento, predicciones_entrenamiento),
        generar_matriz_confusion("test", y_prueba, predicciones_prueba),
    ]

    # Guardar resultados
    guardar_resultados(resultados_evaluacion, "files/output/metrics.json")


if __name__ == "__main__":
    ejecutar_pipeline_completo()