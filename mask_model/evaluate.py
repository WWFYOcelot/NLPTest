
from sklearn.metrics import f1_score

def evaluate(actual_labels, target_vals):
    # Assuming you have the actual labels from your dataset
    # Replace 'actual_labels' with your actual target values from the dataset
    # For example, if your dataset has a 'target' column:
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
