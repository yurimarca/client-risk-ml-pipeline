import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])




##############Function for reporting
def score_model():
    # Load the test data
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    
    # Prepare features and target
    X_test = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = test_data['exited']
    
    # Load the deployed model
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    # Create a figure and axis
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Exited', 'Exited'],
                yticklabels=['Not Exited', 'Exited'])
    
    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Create output directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(model_path, 'confusionmatrix.png'))
    plt.close()
    
    print("Confusion matrix plot saved to confusionmatrix.png")
    
    # Calculate detailed metrics
    f1_score = metrics.f1_score(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    
    # Create classification report
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # Save detailed metrics to CSV
    timestamp = datetime.now().isoformat()
    df_metrics = pd.DataFrame({
        'timestamp': [timestamp],
        'f1_score': [f1_score],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'roc_auc': [roc_auc]
    })
    
    metrics_file = os.path.join(model_path, 'detailed_metrics.csv')
    df_metrics.to_csv(metrics_file, mode='a', header=not os.path.exists(metrics_file), index=False)
    
    # Save classification report
    report_file = os.path.join(model_path, 'classification_report.csv')
    df_report.to_csv(report_file, index=True)
    
    print("\nModel Metrics:")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

def score_redeployed_model():
    # Load the test data
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    
    # Prepare features and target
    X_test = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = test_data['exited']
    
    # Load the redeployed model
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    # Create a figure and axis
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Exited', 'Exited'],
                yticklabels=['Not Exited', 'Exited'])
    
    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Redeployed Model)')
    
    # Create output directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(model_path, 'confusionmatrix2.png'))
    plt.close()
    
    print("Confusion matrix plot for redeployed model saved to confusionmatrix2.png")
    
    # Calculate and print additional metrics
    f1_score = metrics.f1_score(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    
    print(f"\nRedeployed Model Metrics:")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == '__main__':
    score_model()
