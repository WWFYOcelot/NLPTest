from datasets import load_dataset

# Load a dataset; for example, a sentiment analysis dataset
dataset = load_dataset("csv", data_files="nlp-getting-started/train.csv")
print(dataset)
