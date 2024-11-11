import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

# Load the CSV file into a DataFrame
data_path = "nlp-getting-started/train.csv"
df = pd.read_csv(data_path)

# Filter the required columns
df = df[['text', 'target']]  # Use only the `text` and `target` columns

# Load the BERTweet tokenizer
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust `num_labels` based on your task

# Tokenize the text data
def tokenize_function(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

# Convert DataFrame to Dataset and tokenize
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_function, batched=True)

# Rename the 'target' column to 'label' and set the format for PyTorch
dataset = dataset.rename_column("target", "label")
dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])

# Split the dataset (optional) if you need separate train/validation sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Open the file in append mode
with open('tweetResults.txt', 'a') as file:
    # Write text to the file
    for x in (1,2,3):
        for y in (1,3,5,10):
            
            learning_rate_base = x  # The coefficient (2 in this case)
            learning_rate_exponent = -5  # The exponent (e.g., -5 for 2e-5)

            # Calculate the learning rate
            training_args = TrainingArguments(
                output_dir="./",
                evaluation_strategy="epoch",  # Evaluate at the end of each epoch
                save_strategy="no",           # Do not save intermediate checkpoints
                learning_rate=learning_rate_base * 10 ** learning_rate_exponent,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=y,
                weight_decay=0
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )

            trainer.train()
            trainer.save_model(f'./tweet-{x}lr-{y}e')

            trainer.evaluate()

            import numpy as np
            from sklearn.metrics import accuracy_score, classification_report

            # Generate predictions
            predictions = trainer.predict(eval_dataset)
            predicted_labels = np.argmax(predictions.predictions, axis=1)  # Convert logits to class labels
            true_labels = np.array(eval_dataset["label"])

            # Calculate accuracy
            accuracy = accuracy_score(true_labels, predicted_labels)
            file.write(f"Model : TWEET-{x}lr-{y}e")
            file.write(f"Accuracy: {accuracy * 100:.2f}%\n")

            # Optional: View detailed classification report
            file.write(classification_report(true_labels, predicted_labels, target_names=["Class 0", "Class 1"]))            
            file.write("\n=================================================================\n")
        file.write("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
