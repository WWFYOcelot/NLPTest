import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERTweet model and tokenizer, and move model to GPU
bertweet = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-large").to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")

from wordlist import getwords, split_data

def evaluate():
    from sklearn.metrics import f1_score

    # Assuming you have the actual labels from your dataset
    # Replace 'actual_labels' with your actual target values from the dataset
    # For example, if your dataset has a 'target' column:
    actual_labels = [line[4] for line in processed_data]  # Adjust index if needed

    # # Calculate F1 score
    f1 = f1_score(actual_labels, target_vals)
    print(f"F1 Score: {f1:.4f}")

    from sklearn.metrics import precision_score, recall_score

    precision = precision_score(actual_labels, target_vals)
    recall = recall_score(actual_labels, target_vals)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for x in range(len(actual_labels) -1):
        if actual_labels[x] == 1:
            if target_vals[x] == 1:
                tp += 1
            if target_vals[x] == 0:
                fn += 1
        if actual_labels[x] == 0:
            if target_vals[x] == 1:
                fp += 1
            if target_vals[x] == 0:
                tn += 1

    resultratios = [f"tp: {tp/len(actual_labels)} fp: {fp/len(actual_labels)} tn: {tn/len(actual_labels)} fn: {fn/len(actual_labels)}"]
    print(resultratios)
    print(f"Accuracy: {(tp+tn)/len(actual_labels)}")

def compare_with_mask(str1, str2):
    parts = str2.split("<mask>")
    pos = 0

    for part in parts:
        if part:
            pos = str1.find(part, pos)
            if pos == -1:
                return False
            pos += len(part)
    
    return True


file_path = "nlp-getting-started/train.csv"
processed_data = split_data(file_path)
mask_word = getwords(processed_data)

# Mask the word
masked_data = []
for id, line in enumerate(processed_data):
    for mask in mask_word:
        if mask in line[3]:
            masked_line = line[3].replace(mask, tokenizer.mask_token, 1)
            masked_data.append(masked_line.strip())
            break
        else:
            continue
    if id >= len(masked_data):
        masked_data.append(line[3].strip().lower())

print(f"length of procssed_data: {len(processed_data)}")
print(f"length of masked_data: {len(masked_data)}")
# for x in range(5, 15):
#     print(x, processed_data[x], masked_data[x])
predicted_list = []
for id, mask in enumerate(masked_data):
    # Tokenize the masked sentence correctly and move input to GPU
    input_ids = tokenizer(mask, return_tensors="pt")["input_ids"].to(device)

    # Pass through the model to predict the masked word
    with torch.no_grad():
        outputs = bertweet(input_ids)  # Don't move outputs to device
        predictions = outputs.logits

    # Get the index of the masked token ([MASK])
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    # Predict the masked word and move result back to CPU for decoding

    num_elements = mask_token_index.numel()
    if num_elements < 1:
        predicted_list.append("null")
        continue

    predicted_token_id = predictions[0, mask_token_index, :].argmax(dim=-1).cpu().item()
    predicted_word = tokenizer.decode([predicted_token_id])
    predicted_list.append(predicted_word.strip())

# print(mask_word)
target_vals = []
for val in predicted_list:
    val = val.lower()
print(f"length of predicted_list: {len(predicted_list)}")
for id, value in enumerate(predicted_list):
    if value in mask_word:
        target_vals.append(1)
    else:
        target_vals.append(0)

for id, mask in enumerate(masked_data):
    if (id > 20):
        break
    print(f"Target: {target_vals[id]} Predicted word: {predicted_list[id]} --> Sentence: {mask}")

evaluate()

    # print(processed_data[len(processed_data)-1], masked_data[len(masked_data)-1])
    # for id, val in enumerate(masked_data):
    #     if id < 100:
    #         if not compare_with_mask(processed_data[id][3], val):
    #             print(processed_data[id], val)