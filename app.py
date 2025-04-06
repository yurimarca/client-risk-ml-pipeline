from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import diagnostics
import reporting

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    # Get the file path from the request
    file_path = request.json.get('filepath')
    if not file_path:
        return jsonify({'error': 'No file path provided'}), 400
    
    try:
        # Make predictions using the diagnostics module
        predictions = diagnostics.model_predictions(file_path)
        return jsonify({'predictions': predictions}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    # Get the F1 score from the latestscore.txt file
    try:
        with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
            score = float(f.read())
        return jsonify({'f1_score': score}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():        
    # Get summary statistics using the diagnostics module
    try:
        stats = diagnostics.dataframe_summary()
        return jsonify({'summary_statistics': stats}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics_endpoint():        
    # Get all diagnostics
    try:
        # Get execution times
        times = diagnostics.execution_time()
        
        # Get missing data percentages
        missing_data = diagnostics.missing_data()
        
        # Get outdated packages
        packages = diagnostics.outdated_packages_list()
        
        return jsonify({
            'execution_times': times,
            'missing_data_percentages': missing_data,
            'outdated_packages': packages
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
