import os
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from tabulate import tabulate
from utils import initialize_mlflow

# Cargar el conjunto de datos
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# parametros de modelos
random_state =  42
estimators_number = 50

# Lista de modelos a entrenar
models = [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree Regressor", DecisionTreeRegressor(random_state=random_state)),
    ("Random Forest Regressor", RandomForestRegressor(n_estimators=estimators_number, random_state=random_state))
]

#creo un nuevo ambiente de experimentos para no guardarlos en el default
experiment_id = initialize_mlflow("ml_regression_experiments")   

# para imprimir datos en terminal
names = []
mses = []
r2s = []

for name, model in models:
    with mlflow.start_run(experiment_id=experiment_id, run_name=name):
        # Entrenar el modelo
        model.fit(X_train, y_train)
        
        # Predecir en el conjunto de prueba
        y_pred = model.predict(X_test)
        
        # Calcular MSE y R^2
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # para imprimir en terminal
        names.append(name)
        mses.append(mse)
        r2s.append(r2)      
        
        # Registrar parámetros, métricas y modelo en MLflow
        mlflow.log_param("Model", name)
        mlflow.log_metrics({"MSE": mse, "R_2": r2})
        mlflow.sklearn.log_model(model, "model")

        if name == 'Random Forest Regressor':
            mlflow.log_param("n_estimators", estimators_number)
        
        # Generar y guardar gráfico
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Valores Reales")
        plt.ylabel("Predicciones")
        plt.title(f"Valores Reales vs Predicciones: {name}")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        # Guardar la figura
        fig_path = f"figures/{name}_predictions_vs_actuals.png"
        if not os.path.exists("figures"):
            os.makedirs("figures")
        plt.savefig(fig_path)
        plt.close()
        
        # Subir el gráfico a MLflow
        mlflow.log_artifact(fig_path)

# imprimo logs en terminal          
print(tabulate({"model_name": names,"MSE":mses, "R^2":r2s
                    }, headers="keys"))