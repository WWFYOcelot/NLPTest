import pandas as pd

def reduce_dataset(input_file, output_file, target_size=100):
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Calculate the fraction needed to get approximately target_size rows
    frac = target_size / len(df)
    
    # Sample the dataset
    reduced_df = df.sample(frac=frac, random_state=3)
    
    # Save the reduced dataset
    reduced_df.to_csv(output_file, index=False)

# Replace 'input.csv' and 'output.csv' with your actual file paths
reduce_dataset('test.csv', 'labelled2.csv', target_size=100)
