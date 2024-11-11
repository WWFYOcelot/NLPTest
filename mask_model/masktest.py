import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer 

# Load pre-trained BERTweet model and tokenizer
bertweet = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-large")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")

# Original sentence
line = "look at the city last night it was ablaze!"

# Mask the word
mask_word = "ablaze"
if mask_word not in line:
    print(f"Unchanged sentence: {line}")
    exit(0)
masked_line = line.replace(mask_word, tokenizer.mask_token)

print(f"Masked sentence: {masked_line}")

# Tokenize the masked sentence correctly
input_ids = tokenizer(masked_line, return_tensors="pt")["input_ids"]

# Pass through the model to predict the masked word
with torch.no_grad():
    outputs = bertweet(input_ids)  
    predictions = outputs.logits

# Get the index of the masked token ([MASK])
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# Predict the masked word
predicted_token_id = predictions[0, mask_token_index, :].argmax(dim=-1).item()
predicted_word = tokenizer.decode([predicted_token_id])

print(f"Predicted word for [MASK]: {predicted_word}")

#id,keyword,location,text,target