# Client Risk Assessment ML Pipeline

An end-to-end machine learning pipeline for predicting client attrition risk, enabling proactive client management and revenue protection.

## Project Overview

This project implements a dynamic risk assessment system that:
- Predicts client attrition risk using machine learning
- Automatically ingests and processes new data
- Monitors model performance and retrains when needed
- Provides API endpoints for model predictions and diagnostics
- Generates automated reports and visualizations

## System Components

The pipeline consists of five main components:

1. **Data Ingestion**
   - Automatically detects and processes new data files
   - Compiles datasets into a master training dataset
   - Maintains records of ingested files

2. **Model Training & Deployment**
   - Trains logistic regression models for risk prediction
   - Scores models using F1 metric
   - Deploys models to production when performance improves

3. **Diagnostics**
   - Monitors data quality and model performance
   - Tracks execution times for key processes
   - Checks for dependency updates

4. **Reporting**
   - Generates confusion matrix visualizations
   - Provides API endpoints for predictions and diagnostics
   - Creates comprehensive model reports

5. **Process Automation**
   - Automatically checks for new data
   - Monitors for model drift
   - Triggers retraining and redeployment when needed

## Project Structure

```
client-risk-ml-pipeline/
├── config.json              # Configuration settings
├── requirements.txt         # Python dependencies
├── ingestion.py            # Data ingestion script
├── training.py             # Model training script
├── scoring.py              # Model scoring script
├── deployment.py           # Model deployment script
├── diagnostics.py          # System diagnostics script
├── reporting.py            # Report generation script
├── app.py                  # API endpoints
├── wsgi.py                 # API deployment helper
├── apicalls.py             # API testing script
└── fullprocess.py          # Full pipeline automation script
```

## Key Directories

- `/sourcedata/` - Source data for model training
- `/ingesteddata/` - Processed and compiled datasets
- `/models/` - Trained models and scores
- `/production_deployment/` - Production-ready models
- `/testdata/` - Test datasets

## Setup and Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure paths in `config.json`:
   - `input_folder_path`: Source data directory
   - `output_folder_path`: Processed data directory
   - `test_data_path`: Test data directory
   - `output_model_path`: Model storage directory
   - `prod_deployment_path`: Production deployment directory

3. Run the full pipeline:
   ```bash
   python fullprocess.py
   ```

4. Access API endpoints:
   - `/prediction` - Get model predictions
   - `/scoring` - Get model scores
   - `/summarystats` - Get data statistics
   - `/diagnostics` - Get system diagnostics

## Automation

The system runs automatically every 10 minutes via cron job to:
- Check for new data
- Monitor model performance
- Retrain and redeploy models when needed
- Generate updated reports

## License

This project is licensed under the MIT License.

It includes starter code provided by [Udacity](https://www.udacity.com/) for educational purposes as part of the Machine Learning DevOps Engineer Nanodegree.
