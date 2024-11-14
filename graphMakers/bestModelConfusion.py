import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Data from the models
data = [
    {
        "model": "bertweet/best_model_with_scheduler4 : labelled1.csv",
        "TN": 49,
        "FP": 8,
        "FN": 11,
        "TP": 32
    },
    {
        "model": "bertweet/best_model_with_scheduler4 : labelled2.csv",
        "TN": 47,
        "FP": 9,
        "FN": 13,
        "TP": 31
    }
]

# Prepare confusion matrices and save as images
for entry in data:
    tn, fp, fn, tp = entry["TN"], entry["FP"], entry["FN"], entry["TP"]
    matrix = np.array([[tn, fp], [fn, tp]])

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot as an image
    image_filename = f"graphMakers/confusion_matrix_{entry['model'].split(':')[1].strip().replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(image_filename)
    plt.close()  # Close the plot to avoid overlap when generating the next one

    # Print metrics
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    print(f"Model: {entry['model']}")
    print(f"Confusion Matrix:\n{matrix}")
    print(f"Precision: {precision:.3f} Recall: {recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}\n")
