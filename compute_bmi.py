import mlflow

def calculate_bmi(height, weight):
    """ Function to calculate the BMI

    Args:
        height ([float]): [height of a person]
        weight ([float]): [weight of a person]
    """
    return weight / (height/100)**2



if __name__ == "__main__":

    names = ['Bob', 'Lital', 'Simona']
    ages = [18, 32, 44]
    heights = [150, 165, 172]
    weights = [55, 80, 100]


    for i, (height, weight) in enumerate(zip(heights, weights)):   
        
        with mlflow.start_run(run_name = names[i]):
            print(names[i], ages[i], height, weight, calculate_bmi(height, weight))
            
            mlflow.set_tag('age', ages[i])   
            mlflow.log_param('height', height)
            mlflow.log_param('weight', weight)
            mlflow.log_metric("bmi", calculate_bmi(height, weight))