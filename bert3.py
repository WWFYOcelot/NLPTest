from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
import torch

# Load the dataset from CSV
data_files = {
    "train": "path/to/train.csv",  # Change to your actual training data path
    "test": "path/to/test.csv"     # Change to your actual test data path
}
dataset = load_dataset("csv", data_files=data_files)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocess function
def preprocess_function(examples):
    # Tokenize the input text
    tokenized = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    # Add labels
    tokenized['labels'] = examples['target']
    return tokenized

# Apply tokenization and preprocessing
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Filter out any examples where text or target is None
tokenized_datasets = tokenized_datasets.filter(lambda x: x['text'] is not None and x['target'] is not None)

# Set up DataLoader
train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Check for GPU availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1} loss: {loss.item()}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += len(batch['labels'])

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
