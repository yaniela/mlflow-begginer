import mlflow
from mlflow.tracking import MlflowClient

def calculate_bmi(height, weight):
    """ Function to calculate the BMI

    Args:
        height ([float]): [height of a person]
        weight ([float]): [weight of a person]
    """
    return weight / (height/100)**2

# TODO hacer funcion general para para crear los experimentos, pasarle 
# experiment_name como parámetro.

def initialize_mlflow():
    # Initialize MLflow
    experiment_name = "compute_bmi_experiments"  

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
 

if __name__ == "__main__":
    names = ['Bob', 'Lital', 'Simona']
    ages = [18, 32, 44]
    heights = [150, 165, 172]
    weights = [55, 80, 100]  

    experiment_id = initialize_mlflow()
    for i, (height, weight) in enumerate(zip(heights, weights)):   
        
        with mlflow.start_run(experiment_id=experiment_id, run_name = names[i]):
            print(names[i], ages[i], height, weight, calculate_bmi(height, weight))            
            mlflow.set_tag('age', ages[i])   
            mlflow.log_param('height', height)
            mlflow.log_param('weight', weight)
            mlflow.log_metric("bmi", calculate_bmi(height, weight))