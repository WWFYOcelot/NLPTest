from datasets import load_dataset

# Specify the paths to the train and test files
data_files = {
    "train": "nlp-getting-started/train.csv",
    "test": "nlp-getting-started/test.csv"
}

# Load the dataset with separate splits
dataset = load_dataset("csv", data_files=data_files)
print(dataset)  # This should show 'train' and 'test' splits
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize function
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

from torch.utils.data import DataLoader

train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(2000))  # Small subset for example
test_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(2000))

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8)

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Change `num_labels` based on classes

from transformers import AdamW
from transformers import get_scheduler

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

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
