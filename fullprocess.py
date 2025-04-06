import os
import json
import pandas as pd
import numpy as np
import pickle
import time
import subprocess
from datetime import datetime

# Import our custom modules
import training
import scoring
import deployment
import diagnostics
import reporting

# Load configuration
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_model_path = config['output_model_path']

def check_for_new_data():
    """Check if there are new data files that haven't been ingested yet."""
    print(f"Checking for new data in {input_folder_path}...")
    
    # Read the list of already ingested files
    try:
        with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
            ingested_files = set(f.read().splitlines())
        print(f"Currently ingested files: {ingested_files}")
    except FileNotFoundError:
        print("No ingestedfiles.txt found. Starting fresh.")
        ingested_files = set()
    
    # Get list of all CSV files in the input folder
    current_files = set(f for f in os.listdir(input_folder_path) if f.endswith('.csv'))
    print(f"Current files in {input_folder_path}: {current_files}")
    
    # Check for new files
    new_files = current_files - ingested_files
    print(f"New files found: {new_files}")
    
    if new_files:
        print(f"Found {len(new_files)} new file(s) to ingest: {new_files}")
    else:
        print("No new files found.")
    
    return len(new_files) > 0

def check_for_model_drift():
    """Check if the model performance has degraded."""
    # Get the current score from the deployed model
    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
        current_score = float(f.read())
    print(f"Current model score: {current_score}")
    
    # Train a new model with the current data
    print("\nTraining new model for drift detection...")
    new_model = training.train_model()
    
    # Score the new model on test data
    new_score = scoring.score_model(use_deployed_model=False)
    print(f"New model score on test data: {new_score}")
    
    # Compare scores
    if new_score < current_score:
        print(f"Model drift detected! New model performs worse: {new_score} < {current_score}")
        print("This indicates the model's performance has degraded with the new data.")
        return True
    else:
        print(f"No model drift detected. New model performs better or equal: {new_score} >= {current_score}")
        print("This indicates the model is still performing well with the new data.")
        return False

def main():
    print(f"\nRunning full process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check for new data
    print("\nChecking for new data...")
    has_new_data = check_for_new_data()
    
    if has_new_data:
        print("New data found. Proceeding with ingestion and retraining...")
        # Run ingestion
        subprocess.run(['python', 'ingestion.py'], check=True)
        
        # Retrain the model with new data
        print("\nRetraining model with new data...")
        training.train_model()
        
        # Score the new model
        print("\nScoring new model...")
        scoring.score_model()
        
        # Deploy the new model
        print("\nDeploying new model...")
        deployment.store_model_into_pickle()
    else:
        print("No new data found. Checking for model drift...")
    
    # Step 2: Check for model drift
    print("\nChecking for model drift...")
    has_drift = check_for_model_drift()
    
    if has_drift:
        print("Model drift detected. Retraining model...")
        # Retrain the model
        training.train_model()
        
        # Score the new model
        scoring.score_model()
        
        # Deploy the new model
        deployment.store_model_into_pickle()
    else:
        print("No model drift detected.")
    
    # Step 3: Run diagnostics and reporting
    print("\nRunning diagnostics...")
    diagnostics.model_predictions(os.path.join('testdata', 'testdata.csv'))
    diagnostics.dataframe_summary()
    diagnostics.missing_data()
    diagnostics.execution_time()
    diagnostics.outdated_packages_list()
    
    print("\nGenerating reports...")
    reporting.score_model()
    
    print("\nProcess completed successfully!")

if __name__ == '__main__':
    main()


