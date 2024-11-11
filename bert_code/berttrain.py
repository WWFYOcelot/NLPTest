import torch
from torch.utils.data import DataLoader, Dataset
from bert_code.data import dataset

class CustomTextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'attention_mask': self.data[idx]['attention_mask'],
            'labels': torch.tensor(self.data[idx]['target'], dtype=torch.long)
        }

# Initialize the dataset and DataLoader
train_dataset = CustomTextDataset(dataset)  # `dataset` is your list of dictionaries
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Adjust batch_size as needed

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 2 for binary classification
model.to('cuda')  # Move model to GPU if available

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

# Define optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)

# Scheduler for learning rate decay
num_epochs = 10
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
print(scheduler)

from torch.nn import CrossEntropyLoss

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        # Move batch to GPU if available
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate schedule

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}')

from transformers import BertTokenizer

import os
save_directory = 'NLPTest/model-3lr-10e'
os.makedirs(save_directory, exist_ok=True)  # This will create the directory if it doesn't exist

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained(save_directory)
# model.save_pretrained('NLPTest/fine_tuned_bert')
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print("Model storage complete.")

# BertTokenizer.save_pretrained('NLPTest/fine_tuned_bert')
