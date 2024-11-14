import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the CSV file into a DataFrame
data_path = "nlp-getting-started/train.csv"
df = pd.read_csv(data_path)
df = df[['text', 'target']]  # Use only the `text` and `target` columns

# Load the BERTweet tokenizer and model
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the text data
def tokenize_function(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

# Convert DataFrame to Dataset and tokenize
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.rename_column("target", "label")
dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])

# Split the dataset
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Set model parameters for this evaluation run
learning_rate = 1e-5  # Starting learning rate
batch_size = 16       # Batch size
epochs = 3            # Number of epochs
num_entries = len(train_dataset)

# Calculate total steps for the OneCycle scheduler
total_steps = (num_entries // batch_size) * epochs

# Configure training arguments (placeholder scheduler)
training_args = TrainingArguments(
    output_dir="./",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    lr_scheduler_type="constant"  # Placeholder
)

# Initialize Trainer without a scheduler
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda p: {
        "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))
    }
)

total_steps = len(trainer.get_train_dataloader()) * epochs

# Set up the optimizer and the OneCycle scheduler
optimizer = trainer.create_optimizer()
scheduler = OneCycleLR(optimizer, max_lr=learning_rate * 3, total_steps=total_steps)

# Custom training loop with OneCycle scheduler
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    trainer.model.train()
    for step, batch in enumerate(trainer.get_train_dataloader()):
        batch = {k: v.to(trainer.model.device) for k, v in batch.items()}  # Move batch to device
        outputs = trainer.model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        scheduler.step()  # Step the OneCycle scheduler
        optimizer.zero_grad()

    # Evaluate at the end of each epoch
    eval_results = trainer.evaluate()
    print(f"Evaluation results after epoch {epoch + 1}: {eval_results}")

# Save the model
trainer.save_model('bertweet/best_model_with_scheduler4')

# Generate predictions
predictions = trainer.predict(eval_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = np.array(list(eval_dataset["label"]))

# Calculate accuracy and classification report
accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels, target_names=["Class 0", "Class 1"])

# Log results
with open('bertweet/model_with_scheduler_results4.txt', 'w') as file:
    file.write(f"Model: BERTweet with LR={learning_rate}, Batch Size={batch_size}, Epochs={epochs}, Scheduler=OneCycle\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    file.write(report)
    file.write("\n=================================================================\n")
