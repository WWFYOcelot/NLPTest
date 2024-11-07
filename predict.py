from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('NLPTest/fine_tuned_bert')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set model to evaluation mode
model.eval()
model.to('cuda')  # Move to GPU if available

def preprocess_text(text):
    # Tokenize the input text
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'  # Return as PyTorch tensor
    )
    return encoded['input_ids'].to('cuda'), encoded['attention_mask'].to('cuda')

def classify_text(text):
    input_ids, attention_mask = preprocess_text(text)

    # Forward pass through the model
    with torch.no_grad():  # Disable gradient calculation for prediction
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities

from wordlist import split_data
text = split_data("nlp-getting-started/test.csv")
# target_vals = []
# predicted_vals = []
for id, val in enumerate(text):
    if id < 20:
        predicted_class, probabilities = classify_text(val[3])
        print(f"Predicted class: {predicted_class} -> Text: {val[3]}")
        print(f"Class probabilities: {probabilities}")
