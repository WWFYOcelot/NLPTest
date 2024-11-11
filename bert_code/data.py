from transformers import BertTokenizer
from torch.utils.data import Dataset

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import pandas as pd

# Load the dataset from a CSV file
csv_file_path = 'nlp-getting-started/train.csv'  # Update with the path to your CSV file
df = pd.read_csv(csv_file_path)

# Convert the relevant columns into a list of dictionaries
data = df[['text', 'target']].to_dict(orient='records')

# Process data
inputs = []
for item in data:
    encoded = tokenizer.encode_plus(
        item["text"],
        add_special_tokens=True,
        max_length=128,  # Adjust based on your needs
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    inputs.append({
        "input_ids": encoded['input_ids'].squeeze(),
        "attention_mask": encoded['attention_mask'].squeeze(),
        "target": item["target"]
    })

# Create the dataset
dataset = inputs
