import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
from datetime import datetime

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(dataset_path):
    # Load the deployed model
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Read the dataset
    data = pd.read_csv(dataset_path)
    
    # Prepare features
    X = data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions.tolist()

##################Function to get summary statistics
def dataframe_summary():
    # Read the dataset
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Calculate statistics
    stats = []
    for col in numeric_cols:
        stats.extend([
            data[col].mean(),
            data[col].median(),
            data[col].std()
        ])
    
    return stats

##################Function to check for missing data
def missing_data():
    # Read the dataset
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    
    # Calculate percentage of NA values in each column
    na_percentages = (data.isna().sum() / len(data) * 100).tolist()
    
    return na_percentages

##################Function to get timings
def execution_time():
    # Time ingestion.py
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - start_time
    
    # Time training.py
    start_time = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - start_time
    
    return [ingestion_time, training_time]

##################Function to check dependencies
def outdated_packages_list():
    # Get installed packages
    installed = subprocess.check_output(['pip', 'list', '--format=json']).decode()
    installed = json.loads(installed)
    
    # Get outdated packages
    outdated = subprocess.check_output(['pip', 'list', '--outdated', '--format=json']).decode()
    outdated = json.loads(outdated)
    
    # Create a dictionary of current and latest versions
    package_info = []
    for pkg in installed:
        current_version = pkg['version']
        latest_version = next((item['latest_version'] for item in outdated 
                             if item['name'] == pkg['name']), current_version)
        package_info.append({
            'name': pkg['name'],
            'current_version': current_version,
            'latest_version': latest_version
        })
    
    return package_info


if __name__ == '__main__':
    # Test predictions
    test_data = os.path.join(test_data_path, 'testdata.csv')
    predictions = model_predictions(test_data)
    print(f"Model predictions: {predictions[:5]}...")  # Print first 5 predictions
    
    # Test summary statistics
    stats = dataframe_summary()
    print(f"Summary statistics: {stats}")
    
    # Test missing data
    na_percentages = missing_data()
    print(f"Missing data percentages: {na_percentages}")
    
    # Test execution time
    times = execution_time()
    print(f"Execution times - Ingestion: {times[0]:.2f}s, Training: {times[1]:.2f}s")
    
    # Test dependency check
    packages = outdated_packages_list()
    print("Package versions:")
    for pkg in packages[:5]:  # Print first 5 packages
        print(f"{pkg['name']}: {pkg['current_version']} -> {pkg['latest_version']}")





    
