import numpy as np
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
    print("Score: %s" % score)
    return lr, score

def support_vector_machine(X, y):
    """ 
    Build a support vector classification model
    """
    svc = svm.SVC()
    svc.fit(X, y)
    score = svc.score(X, y)
    print("Score: %s" % score)
    return svc, score


if __name__ == "__main__":
    
    X = np.array([10, 20, 200, 250, 300, 30]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    
    lr_model, lr_score = logistic_regression(X, y)    
    svc_model, svc_score = support_vector_machine(X, y)

    models = [lr_model, svc_model]
    scores = [lr_score, svc_score]
    model_names = ["Logistic regression", "Support vector machine"]

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    for i, (model, score) in enumerate(zip(models, scores)):   
        with mlflow.start_run(run_name = model_names[i]):
            mlflow.log_metric("score", score)
            mlflow.sklearn.log_model(model, "model")
            print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    