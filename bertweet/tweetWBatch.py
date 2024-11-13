import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
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

# Set dataset size and epochs
num_entries = len(train_dataset)
epochs_list = [3, 4, 5]
batch_sizes = [16, 32, 64]
learning_rates = [1e-5, 2e-5, 3e-5]

# Open a file to log results
with open('bertweet/tweetWBatchesResults2.txt', 'a') as file:
    for batch_size in batch_sizes:
        for lr in learning_rates:
            for epochs in epochs_list:
                # Calculate warmup steps (5% of total steps)
                total_steps = (num_entries // batch_size) * epochs
                warmup_steps = int(0.05 * total_steps)

                # Configure training arguments
                training_args = TrainingArguments(
                    output_dir="./",
                    evaluation_strategy="epoch",
                    save_strategy="no",
                    learning_rate=lr,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    num_train_epochs=epochs,
                    warmup_steps=warmup_steps,
                    weight_decay=0.01
                )

                # Initialize Trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=lambda p: {
                        "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))
                    }
                )

                # Train the model
                trainer.train()
                
                # Save the model
                model_dir = f'./tweet-lr{lr}-bs{batch_size}-ep{epochs}'
                trainer.save_model(model_dir)

                # Evaluate the model
                eval_results = trainer.evaluate()
                
                # Generate predictions
                predictions = trainer.predict(eval_dataset)
                predicted_labels = np.argmax(predictions.predictions, axis=1)
                true_labels = np.array(list(eval_dataset["label"]))

                # Calculate accuracy and classification report
                accuracy = accuracy_score(true_labels, predicted_labels)
                report = classification_report(true_labels, predicted_labels, target_names=["Class 0", "Class 1"])

                # Log results to the file
                file.write(f"Model: BERTweet with LR={lr}, Batch Size={batch_size}, Epochs={epochs}, Warmup Steps={warmup_steps}\n")
                file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
                file.write(report)
                file.write("\n=================================================================\n")
            file.write("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
