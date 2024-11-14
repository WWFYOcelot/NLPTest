from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import torch
import pandas as pd

scores = []

# Open the file in append mode
with open('bertweet/bestModelLabelled.txt', 'a') as file:
    vals = 0
    model_name = 'bertweet/best_model_with_scheduler4'
    
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for z in (1,2):
        # Load dataset
        data = pd.read_csv(f"nlp-getting-started/labelled{z}.csv")  # Ensure 'text' and 'target' columns
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

        # Initialize model for each fold
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for text in texts:
                input_ids, attention_mask = preprocess_text(text)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
                val_preds.append(pred)

        # Compute the confusion matrix
        cm = confusion_matrix(labels, val_preds)

        # Extract TP, TN, FP, FN from the confusion matrix
        TN, FP, FN, TP = cm.ravel()

        file.write(f"{model_name} : labelled{z}.csv\n")
        # Print the values
        file.write(f'True Negative: {TN}\n')
        file.write(f'False Positive: {FP}\n')
        file.write(f'False Negative: {FN}\n')
        file.write(f'True Positive: {TP}\n')

        # Calculate F1 score for this fold
        f1 = f1_score(labels, val_preds)
        vals += f1
        recall = recall_score(labels, val_preds)
        precision = precision_score(labels, val_preds)
        file.write(f"Precision: {precision} Recall: {recall}\n")
        file.write(f"F1 Score: {f1}\n")