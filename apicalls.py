import requests
import json
import os

# Base URL for the API
BASE_URL = 'http://localhost:8000'

def test_prediction():
    """Test the prediction endpoint"""
    try:
        # Prepare the request
        data = {'filepath': os.path.join('testdata', 'testdata.csv')}
        
        # Make the request
        response = requests.post(f'{BASE_URL}/prediction', json=data)
        
        # Print results
        print("\nPrediction Endpoint:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.json()
    except Exception as e:
        print(f"Error testing prediction endpoint: {str(e)}")
        return None

def test_scoring():
    """Test the scoring endpoint"""
    try:
        # Make the request
        response = requests.get(f'{BASE_URL}/scoring')
        
        # Print results
        print("\nScoring Endpoint:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.json()
    except Exception as e:
        print(f"Error testing scoring endpoint: {str(e)}")
        return None

def test_summary_stats():
    """Test the summary statistics endpoint"""
    try:
        # Make the request
        response = requests.get(f'{BASE_URL}/summarystats')
        
        # Print results
        print("\nSummary Statistics Endpoint:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.json()
    except Exception as e:
        print(f"Error testing summary stats endpoint: {str(e)}")
        return None

def test_diagnostics():
    """Test the diagnostics endpoint"""
    try:
        # Make the request
        response = requests.get(f'{BASE_URL}/diagnostics')
        
        # Print results
        print("\nDiagnostics Endpoint:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.json()
    except Exception as e:
        print(f"Error testing diagnostics endpoint: {str(e)}")
        return None

def main():
    """Test all API endpoints and save combined results"""
    results = {
        'predictions': test_prediction(),
        'scoring': test_scoring(),
        'summary_stats': test_summary_stats(),
        'diagnostics': test_diagnostics()
    }
    
    # Save combined results to a file
    with open('apireturns.txt', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nAll results saved to apireturns.txt")

def test_redeployed_model():
    """Test all API endpoints with redeployed model and save combined results"""
    # First, ensure the model is redeployed
    import deployment
    deployment.store_model_into_pickle()
    
    results = {
        'predictions': test_prediction(),
        'scoring': test_scoring(),
        'summary_stats': test_summary_stats(),
        'diagnostics': test_diagnostics()
    }
    
    # Save combined results to a file
    with open('apireturns2.txt', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nAll results for redeployed model saved to apireturns2.txt")

if __name__ == '__main__':
    main()
    test_redeployed_model()



