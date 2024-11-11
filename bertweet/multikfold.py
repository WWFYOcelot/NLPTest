from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import torch
import pandas as pd
from tqdm import tqdm

# Open file for writing results
with open('bertweet/kfoldvalresults.txt', 'a') as file:
    # Loop over different learning rates and epoch values
    for x in (1, 2, 3):  # Learning rates or similar parameter variations
        for y in (1, 3, 5, 10):  # Epochs or similar parameter variations
            # Define model name with learning rate and epochs
            model_name = f'bertweet/tweet-{x}lr-{y}e'

            # Load the BERTweet tokenizer for the specific model
            tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')

            # Define device (GPU if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load dataset
            data = pd.read_csv("nlp-getting-started/train.csv")  # Ensure 'text' and 'target' columns
            texts = data["text"].values
            labels = data["target"].values

            # Function to preprocess text and tokenize in batches
            def preprocess_text(texts):
                return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

            # Different fold configurations
            folds = [3, 5, 10]
            file.write(f"Model: TWEET-{x}lr-{y}e\n")

            # Initialize model once before k-fold loop
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

            for k in folds:
                print(f"Evaluating TWEET-{x}lr-{y}e with k={k} folds:")
                kf = KFold(n_splits=k)
                f1_scores = []

                for train_index, val_index in kf.split(texts):
                    # Split data
                    train_texts, val_texts = texts[train_index], texts[val_index]
                    train_labels, val_labels = labels[train_index], labels[val_index]

                    # Validation loop
                    model.eval()  # Set the model to evaluation mode
                    val_preds = []
                    with torch.no_grad():
                        for i in range(0, len(val_texts), 16):  # Batch size = 16
                            batch_texts = val_texts[i:i + 16]
                            batch_labels = val_labels[i:i + 16]

                            batch_texts = list(batch_texts)  # Ensure it's a list of strings

                            # Tokenize the batch
                            batch_inputs = preprocess_text(batch_texts)
                            batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}

                            # Forward pass
                            outputs = model(**batch_inputs)
                            logits = outputs.logits
                            preds = torch.argmax(logits, dim=1).cpu().numpy()

                            val_preds.extend(preds)

                    # Calculate F1 score for this fold
                    fold_f1 = f1_score(val_labels, val_preds)
                    f1_scores.append(fold_f1)
                    file.write(f"F1 Score for fold {len(f1_scores)}: {fold_f1}\n")

                # Write average F1 score across all folds
                average_f1 = sum(f1_scores) / len(f1_scores)
                file.write(f"Average F1 Score for k={k}: {average_f1}\n")

            file.write("===================================================\n")
