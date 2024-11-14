import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

# Load the CSV file into a DataFrame
data_path = "nlp-getting-started/train.csv"
df = pd.read_csv(data_path)
df = df[['text', 'target']]  # Use only the `text` and `target` columns

# Load the BERTweet tokenizer and model
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

# Function to initialize Trainer and evaluate with different strategies
def run_experiment(training_args, optimizer=None, scheduler=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Initialize Trainer with optimizer and scheduler
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))
        },
        optimizers=(optimizer, scheduler) if optimizer else None
    )
    trainer.train()

    # Evaluate and save the model
    eval_results = trainer.evaluate()
    predictions = trainer.predict(eval_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = np.array(list(eval_dataset["label"]))

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=["Class 0", "Class 1"])

    return accuracy, report

# Save results function
def save_results(experiment_name, accuracy, report):
    with open(f'bertweet/experiment_results.txt', 'a') as file:
        file.write(f"Experiment: {experiment_name}\n")
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        file.write(report)
        file.write("\n==============================\n")

# Experiment 1: Test with OneCycleLR Scheduler
def experiment_onecycle_lr():
    # Configure the OneCycleLR scheduler
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Manually calculate the number of steps
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * epochs
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate * 3, total_steps=total_steps)

    training_args = TrainingArguments(
        output_dir="./",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_steps=int(0.05 * total_steps),  # 5% of total steps for warmup
        weight_decay=0.01,
        lr_scheduler_type="constant",  # Set to constant as we manually control the scheduler
    )

    accuracy, report = run_experiment(training_args, optimizer, scheduler)
    save_results("OneCycleLR", accuracy, report)

# Experiment 2: Test with Cosine Annealing Scheduler
def experiment_cosine():
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * epochs

    training_args = TrainingArguments(
        output_dir="./",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_steps=int(0.05 * total_steps),
        weight_decay=0.01,
        lr_scheduler_type="cosine_with_restarts",  # Cosine Annealing Scheduler
    )
    
    accuracy, report = run_experiment(training_args)
    save_results("CosineAnnealing", accuracy, report)

# Experiment 3: Test with Linear Scheduler
def experiment_linear():
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * epochs

    training_args = TrainingArguments(
        output_dir="./",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_steps=int(0.05 * total_steps),
        weight_decay=0.01,
        lr_scheduler_type="linear",  # Linear Scheduler
    )
    
    accuracy, report = run_experiment(training_args)
    save_results("LinearScheduler", accuracy, report)

# Experiment 4: Test with More Epochs
def experiment_more_epochs():
    global epochs
    epochs = 5  # Increase epochs to 5

    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * epochs

    training_args = TrainingArguments(
        output_dir="./",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_steps=int(0.05 * total_steps),
        weight_decay=0.01,
    )

    accuracy, report = run_experiment(training_args)
    save_results("MoreEpochs", accuracy, report)

# Experiment 5: Test with Different Batch Size
def experiment_different_batch_size():
    global batch_size
    batch_size = 32  # Increase batch size to 32

    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * epochs

    training_args = TrainingArguments(
        output_dir="./",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_steps=int(0.05 * total_steps),
        weight_decay=0.01,
    )

    accuracy, report = run_experiment(training_args)
    save_results("DifferentBatchSize", accuracy, report)

# Running all experiments
if __name__ == "__main__":
    experiment_onecycle_lr()
    experiment_cosine()
    experiment_linear()
    experiment_more_epochs()
    experiment_different_batch_size()
