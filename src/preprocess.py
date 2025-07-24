import pandas as pd
import sys,os
import yaml

params = yaml.safe_load(open("params.yaml"))["preprocess"]

def preprocess_data(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Perform preprocessing steps
    df.drop_duplicates(inplace=True)

    # Save the preprocessed data
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    input_file = params["input"]
    output_file = params["output"]
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        sys.exit(1)
    
    preprocess_data(input_file, output_file)