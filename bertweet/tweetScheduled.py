from transformers import get_linear_schedule_with_warmup
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the CSV file into a DataFrame
data_path = "nlp-getting-started/train.csv"
df = pd.read_csv(data_path)

# Filter the required columns
df = df[['text', 'target']]  # Use only the `text` and `target` columns

# Load the BERTweet tokenizer
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the text data
def tokenize_function(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

# Convert DataFrame to Dataset and tokenize
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_function, batched=True)

# Rename the 'target' column to 'label' and set the format for PyTorch
dataset = dataset.rename_column("target", "label")
dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])

# Split the dataset into train/validation sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

learning_rate = 1e-5
num_epochs = 3
weight_decay = 0.01
warmup_steps = 10
batch_size = 128

# Define training arguments with warmup and cosine decay
training_args = TrainingArguments(
    output_dir="./",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train and evaluate the model
trainer.train()
trainer.save_model(f'./tweet-lr-{learning_rate}-e-{num_epochs}-wd-{weight_decay}-steps{warmup_steps}')

# Generate predictions
predictions = trainer.predict(eval_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = np.array(list(eval_dataset["label"]))

# Calculate accuracy and classification report
accuracy = accuracy_score(true_labels, predicted_labels)

# Save results
with open('bertweet/scheduledTweetResults.txt', 'a') as file:
    file.write(f"Model: BERTweet with LR={learning_rate}, Epochs={num_epochs}, Weight Decay={weight_decay}, Warmup Steps={warmup_steps}, Batch Size={batch_size}\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    file.write(classification_report(true_labels, predicted_labels, target_names=["Class 0", "Class 1"]))            
    file.write("\n=================================================================\n")
    print("Data saved")
