from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from datetime import datetime



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model(use_deployed_model=False, dataset_path=None):
    # Load the data
    if use_deployed_model:
        # Always use test data for drift detection
        data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
        print("Using test data for drift detection...")
    else:
        if dataset_path:
            data = pd.read_csv(dataset_path)
            print(f"Using dataset from {dataset_path} for scoring...")
        else:
            data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
            print("Using test data for scoring...")
    
    # Prepare features and target
    X = data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = data['exited']
    
    # Load the model from either deployment or training directory
    if use_deployed_model:
        model_path = config['prod_deployment_path']
        print("Using deployed model for scoring...")
    else:
        model_path = config['output_model_path']
        print("Using trained model for scoring...")
    
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate F1 score
    f1_score = metrics.f1_score(y, y_pred)
    
    # Save the score with timestamp to CSV
    timestamp = datetime.now().isoformat()
    df_scores = pd.DataFrame({
        'timestamp': [timestamp],
        'f1_score': [f1_score],
        'model_type': ['deployed' if use_deployed_model else 'trained'],
        'dataset': [os.path.basename(dataset_path) if dataset_path else 'testdata.csv']
    })
    
    scores_file = os.path.join(model_path, 'model_scores.csv')
    df_scores.to_csv(scores_file, mode='a', header=not os.path.exists(scores_file), index=False)
    
    # Also save the latest score to txt for backward compatibility
    if not use_deployed_model:
        with open(os.path.join(model_path, 'latestscore.txt'), 'w') as f:
            f.write(str(f1_score))
    
    print(f"Model F1 score: {f1_score}")
    return f1_score


if __name__ == '__main__':
    score_model()

