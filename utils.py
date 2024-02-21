
import mlflow
from mlflow.tracking import MlflowClient


def initialize_mlflow(experiments_name):
    # Initialize MLflow
    experiment_name = experiments_name  

    # Provide uri and connect to your tracking server
    mlflow.set_tracking_uri('http://127.0.0.1:5000/')

    # Initialize MLflow client
    client = MlflowClient()

    # If experiment doesn't exist then it will create new
    # else it will take the experiment id and will use to to run the experiments
    try:
        # Create experiment 
        experiment_id = client.create_experiment(experiment_name)
        return experiment_id
    except:
        # Get the experiment id if it already exists
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        return experiment_id