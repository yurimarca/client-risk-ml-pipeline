import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
from datetime import datetime
from scipy.stats import ks_2samp

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

def data_drift_check():
    """Check for data drift between the last ingested data and the current data."""
    # Get the paths
    current_data_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    prod_data_path = os.path.join(prod_deployment_path, 'finaldata.csv')
    
    if not os.path.exists(prod_data_path):
        print("No previous data found for drift comparison")
        return None
    
    # Read the datasets
    df_current = pd.read_csv(current_data_path)
    df_prod = pd.read_csv(prod_data_path)
    
    # Get numeric columns
    numeric_cols = df_current.select_dtypes(include=[np.number]).columns
    
    # Initialize drift report
    drift_report = {}
    
    # Compare distributions for each numeric column
    for col in numeric_cols:
        if col in df_prod.columns:
            stat, p_value = ks_2samp(df_prod[col], df_current[col])
            drift_report[col] = {
                'statistic': stat,
                'p_value': p_value,
                'drifted': p_value < 0.05
            }
    
    # Save drift report to CSV
    timestamp = datetime.now().isoformat()
    df_drift = pd.DataFrame({
        'timestamp': [timestamp],
        'column': list(drift_report.keys()),
        'statistic': [drift_report[col]['statistic'] for col in drift_report],
        'p_value': [drift_report[col]['p_value'] for col in drift_report],
        'drifted': [drift_report[col]['drifted'] for col in drift_report]
    })
    
    drift_file = os.path.join(dataset_csv_path, 'data_drift.csv')
    df_drift.to_csv(drift_file, mode='a', header=not os.path.exists(drift_file), index=False)
    
    return drift_report

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





    
