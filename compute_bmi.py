import mlflow
from tabulate import tabulate

def calculate_bmi(height, weight):
    """ Funcion para calcular el indice de masa corporal
     en relación a la estatura y el peso.

    Args:
        height ([float]): [estatura de la persona]
        weight ([float]): [peso de la persona]
    """
    return weight / (height/100)**2
 

if __name__ == "__main__":

    # datos que van a ser tipo etiquetas
    names = ['Bob', 'Lital', 'Simona']
    ages = [18, 32, 56]

    # parámetros de entrada de la funcion que calcula el BMI
    heights = [150, 165, 123]
    weights = [55, 80, 50]

    # Para almacenar el resultado y mostrarlo en la terminal
    bmis = []  #BMIs a calcular

    # Provide uri and connect to your tracking server
    mlflow.set_tracking_uri('http://127.0.0.1:5000/')

    for i, (height, weight) in enumerate(zip(heights, weights)):   
        
        with mlflow.start_run(run_name = names[i]):           
            bmis.append(calculate_bmi(height, weight))
            mlflow.set_tag('age', ages[i])   
            mlflow.log_param('height', height)
            mlflow.log_param('weight', weight)
            mlflow.log_metric("bmi", bmis[i])

    
    # imprimo logs en terminal       
    print(tabulate({"name": names,"age":ages,"height":heights,
                    "weight":weights,"bmi":bmis}, headers="keys"))
    
