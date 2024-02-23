import numpy as np
from tabulate import tabulate
from utils import initialize_mlflow

from sklearn.linear_model import LogisticRegression
from sklearn import svm

import mlflow
import mlflow.sklearn


def logistic_regression(X, y):
    """ 
    Build a logistic regression model
    """
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    #print("Score: %s" % score)
    return lr, score

def support_vector_machine(X, y):
    """ 
    Build a support vector classification model
    """
    svc = svm.SVC()
    svc.fit(X, y)
    score = svc.score(X, y)
    #print("Score: %s" % score)
    return svc, score


if __name__ == "__main__":
    
    X = np.array([10, 20, 200, 250, 300, 30]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])

    # ejecuto los dos algoritos para obtener los modelos y los valores de la métrica score
    lr_model, lr_score = logistic_regression(X, y)    
    svc_model, svc_score = support_vector_machine(X, y)

   
    models = [lr_model, svc_model]
    scores = [lr_score, svc_score]
    model_names = ["Logistic regression", "Support vector machine"]

    #creo un nuevo ambiente de experimentos para no guardarlos en el default
    experiment_id = initialize_mlflow("ml_classification_experiments")  
    runs_ids=[] 
    
    # almaceno tracks en  mlflow
    for i, (model, score) in enumerate(zip(models, scores)):   
        with mlflow.start_run(experiment_id=experiment_id, run_name = model_names[i]):
            mlflow.log_metric("score", score)
            mlflow.sklearn.log_model(model, "model")
            runs_ids.append(mlflow.active_run().info.run_uuid)

    # imprimo logs en terminal          
    print(tabulate({"model_name": model_names,"score":scores, "experiment_run_id":runs_ids
                    }, headers="keys"))
    