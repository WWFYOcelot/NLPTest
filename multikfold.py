from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import torch
import pandas as pd
from tqdm import tqdm

# Load the fine-tuned model and tokenizer
model_name = 'NLPTest/model-3lr-10e'
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
        # model.train()

        # Training (Placeholder - include your training loop if needed)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        # epochs = 2
        # for epoch in range(epochs):
        #     for text, label in zip(train_texts, train_labels):
        #         input_ids, attention_mask = preprocess_text(text)
        #         label = torch.tensor([label]).to(device)
        #         optimizer.zero_grad()
        #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        #         loss = outputs.loss
        #         loss.backward()
        #         optimizer.step()

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
        print(f"F1 Score for fold {len(f1_scores)}: {fold_f1}")

    # Average F1 score across all folds
    average_f1 = sum(f1_scores) / len(f1_scores)
    print(f"Average F1 Score for k={k}: {average_f1}")
