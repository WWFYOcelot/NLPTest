import matplotlib.pyplot as plt
import numpy as np

# Data for precision and recall
models = [
    'TWEET-1lr-1e', 'TWEET-1lr-3e', 'TWEET-1lr-5e', 'TWEET-1lr-10e',
    'TWEET-2lr-1e', 'TWEET-2lr-3e', 'TWEET-2lr-5e', 'TWEET-2lr-10e',
    'TWEET-3lr-1e', 'TWEET-3lr-3e', 'TWEET-3lr-5e', 'TWEET-3lr-10e'
]
precision_class_0 = [0.81, 0.82, 0.83, 0.83, 0.82, 0.83, 0.82, 0.82, 0.83, 0.84, 0.82, 0.82]
precision_class_1 = [0.86, 0.87, 0.81, 0.81, 0.83, 0.79, 0.78, 0.80, 0.77, 0.79, 0.78, 0.78]

recall_class_0 = [0.90, 0.91, 0.85, 0.86, 0.87, 0.83, 0.82, 0.84, 0.81, 0.83, 0.82, 0.83]
recall_class_1 = [0.74, 0.76, 0.77, 0.77, 0.76, 0.78, 0.77, 0.77, 0.80, 0.80, 0.78, 0.77]

# Bar positions
x = np.arange(len(models))
width = 0.35  # Bar width

# Colors for precision and recall (matching colors)
precision_color_0 = '#5C9BCF'  # Blue for Class 0 Precision
precision_color_1 = '#1F77B4'  # Lighter blue for Class 1 Precision
recall_color_0 = '#F68F70'     # Red for Class 0 Recall
recall_color_1 = '#E14B32'     # Lighter red for Class 1 Recall

# Plotting the overlayed bars for precision and recall
plt.figure(figsize=(12, 6))

# Overlayed precision for Class 0 and Class 1 (Both start from y=0)
plt.bar(x, precision_class_0, width=width, label='Class 0 Precision', color=precision_color_0, alpha=0.7)
plt.bar(x, precision_class_1, width=width, label='Class 1 Precision', color=precision_color_1, alpha=0.7)

# Overlayed recall for Class 0 and Class 1 (Both start from y=0)
plt.bar(x + width, recall_class_0, width=width, label='Class 0 Recall', color=recall_color_0, alpha=0.7)
plt.bar(x + width, recall_class_1, width=width, label='Class 1 Recall', color=recall_color_1, alpha=0.7)

# Labels and title
plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Precision & Recall Scores of BERTweet Models Fine-Tuned on the Disaster Tweets Dataset ')
plt.xticks(x + width / 2, models, rotation=45, ha='right')
plt.ylim(0.7, 0.95)
plt.legend()

# Show plot
plt.tight_layout()
plt.savefig("graphMakers/tweetHoldOutPR")
plt.show()
