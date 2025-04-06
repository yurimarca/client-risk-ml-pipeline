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
   - Automatically detects and processes new data files from `/sourcedata/`
   - Compiles datasets into a master training dataset (`finaldata.csv`)
   - Maintains records of ingested files (`ingestedfiles.txt`)

2. **Model Training & Deployment**
   - Trains logistic regression models for risk prediction
   - Scores models using F1 metric
   - Deploys models to production when performance improves
   - Automatically detects model drift and triggers retraining

3. **Diagnostics**
   - Monitors data quality and model performance
   - Tracks execution times for key processes
   - Checks for dependency updates
   - Provides comprehensive system diagnostics

4. **Reporting**
   - Generates confusion matrix visualizations (`confusionmatrix.png`)
   - Provides API endpoints for predictions and diagnostics
   - Creates comprehensive model reports
   - Logs all API responses (`apireturns.txt`)

5. **Process Automation**
   - Automatically checks for new data every 3 minutes
   - Monitors for model drift using test data
   - Triggers retraining and redeployment when needed
   - Maintains detailed logs of all operations (`cron.log`)

## Project Structure

```
client-risk-ml-pipeline/
├── config.json              # Configuration settings
├── requirements.txt         # Python dependencies
├── setup_cron.sh           # Cron job setup script
├── cronjob.txt             # Cron job configuration
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

- `/sourcedata/` - Source data for model training (dataset3.csv, dataset4.csv)
- `/ingesteddata/` - Processed and compiled datasets (finaldata.csv)
- `/models/` - Trained models and scores (trainedmodel.pkl, latestscore.txt)
- `/production_deployment/` - Production-ready models and records
- `/testdata/` - Test datasets for model evaluation

## Setup and Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure paths in `config.json`:
   ```json
   {
     "input_folder_path": "sourcedata",
     "output_folder_path": "ingesteddata",
     "test_data_path": "testdata",
     "output_model_path": "models",
     "prod_deployment_path": "production_deployment"
   }
   ```

3. Set up the cron job:
   ```bash
   chmod +x setup_cron.sh
   ./setup_cron.sh
   ```

4. Run the full pipeline manually:
   ```bash
   python fullprocess.py
   ```

5. Access API endpoints:
   - `/prediction` - Get model predictions
   - `/scoring` - Get model scores
   - `/summarystats` - Get data statistics
   - `/diagnostics` - Get system diagnostics

## Automation

The system runs automatically every 3 minutes via cron job to:
- Check for new data in `/sourcedata/`
- Monitor model performance using test data
- Retrain and redeploy models when drift is detected
- Generate updated reports and diagnostics

The cron job is configured in `cronjob.txt` and can be set up using `setup_cron.sh`.

## Monitoring

The system maintains several log files:
- `cron.log` - Detailed logs of automated runs
- `apireturns.txt` - API response logs for the current model
- `apireturns2.txt` - API response logs for the redeployed model
- `confusionmatrix.png` - Model performance visualization for the current model
- `confusionmatrix2.png` - Model performance visualization for the redeployed model

## License

This project is licensed under the MIT License.

It includes starter code provided by [Udacity](https://www.udacity.com/) for educational purposes as part of the Machine Learning DevOps Engineer Nanodegree.
