import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    # Initialize an empty list to store DataFrames
    dfs = []
    ingested_files = []
    
    # Get all CSV files in the input folder
    for file in os.listdir(input_folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(input_folder_path, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
            ingested_files.append(file)
    
    # Combine all DataFrames
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates
        final_df = final_df.drop_duplicates()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Save the final dataset
        final_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)
        
        # Save the list of ingested files
        with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
            f.write('\n'.join(ingested_files))
        
        print(f"Successfully ingested {len(ingested_files)} files")
        print(f"Final dataset shape: {final_df.shape}")
    else:
        print("No CSV files found in the input directory")


if __name__ == '__main__':
    merge_multiple_dataframe()
