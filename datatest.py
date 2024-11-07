import pandas as pd

# Load the CSV file
data = pd.read_csv("nlp-getting-started/sample_submission.csv")

# Store the 'target' column in an array
target_values = data["target"].values
print(target_values[0:20])
