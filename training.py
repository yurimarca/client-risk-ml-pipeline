from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    # Read the training data
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    
    # Prepare features and target
    X = data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = data['exited']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LogisticRegression(
        C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=100,
        multi_class='auto', n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        warm_start=False
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Create output directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save the trained model
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model trained and saved to {os.path.join(model_path, 'trainedmodel.pkl')}")
    return model


if __name__ == '__main__':
    train_model()

