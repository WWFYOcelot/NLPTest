from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import torch
import pandas as pd
from tqdm import tqdm

# Load the fine-tuned model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
data = pd.read_csv("nlp-getting-started/train.csv")  # Ensure 'text' and 'target' columns
texts = data["text"].values
labels = data["target"].values

# Function to preprocess text
def preprocess_text(text):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoded['input_ids'].to(device), encoded['attention_mask'].to(device)


# Open file for writing results
with open('bert_code/kfoldvalresults.txt', 'a') as file:
    # Loop over different learning rates and epoch values
    for x in (1, 2, 3):  # Learning rates or similar parameter variations
        for y in (1, 3, 5, 10):  # Epochs or similar parameter variations
            # Define model name with learning rate and epochs
            model_name = f'bertweet/tweet-{x}lr-{y}e'
            file.write(f"Model : BERT-{x}lr-{y}e\n")
            # Different fold configurations
            folds = [3, 5, 10]

            for k in folds:
                print(f"\nEvaluating with k={k} folds:")
                kf = KFold(n_splits=k)
                f1_scores = []

                for train_index, val_index in kf.split(texts):
                    # Split data
                    train_texts, val_texts = texts[train_index], texts[val_index]
                    train_labels, val_labels = labels[train_index], labels[val_index]
                    
                    # Initialize model for each fold
                    model = BertForSequenceClassification.from_pretrained(model_name).to(device)

                    # Validation
                    model.eval()
                    val_preds = []
                    with torch.no_grad():
                        for text in val_texts:
                            input_ids, attention_mask = preprocess_text(text)
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = outputs.logits
                            pred = torch.argmax(logits, dim=1).item()
                            val_preds.append(pred)

                    # Calculate F1 score for this fold
                    fold_f1 = f1_score(val_labels, val_preds)
                    f1_scores.append(fold_f1)
                    file.write(f"F1 Score for fold {len(f1_scores)}: {fold_f1}\n")

                # Average F1 score across all folds
                average_f1 = sum(f1_scores) / len(f1_scores)
                file.write(f"Average F1 Score for k={k}: {average_f1}\n")
        file.write("===============================================")
    file.write("|||||||||||||||||||||||||||||||||||||||||||||||||||")
