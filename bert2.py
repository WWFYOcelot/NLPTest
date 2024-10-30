from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
import torch

# Load a pre-trained tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and include labels
def preprocess_function(examples):
    # Tokenize text and ensure output as tensors
    tokenized = tokenizer(
        examples['text'], padding="max_length", truncation=True, max_length=512
    )
    # Add labels as tensors
    tokenized['labels'] = examples['target']
    return tokenized

# Specify the paths to the train and test files
data_files = {
    "train": "nlp-getting-started/train.csv",
    "test": "nlp-getting-started/test.csv"
}

# Load the dataset with separate splits
dataset = load_dataset("csv", data_files=data_files)

# Check and remove examples with None in 'text' or 'labels'
dataset = dataset.filter(lambda example: example['text'] is not None and example['target'] is not None)
print(dataset)  # This should show 'train' and 'test' splits

# Apply tokenization
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Create train and test datasets
train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(2000))  # Small subset for example
test_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(2000))

# Custom collate function to ensure batch formatting
def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch])
    }

# Define dataloaders with collate function
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Change `num_labels` based on classes

# Setup optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Check for CUDA
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1} loss: {loss.item()}")

# Evaluation loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += len(batch['labels'])

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
