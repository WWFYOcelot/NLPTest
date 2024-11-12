import matplotlib.pyplot as plt
import numpy as np

# Data for models
models = [
    "tweet-1lr-1e", "tweet-1lr-3e", "tweet-1lr-5e", "tweet-1lr-10e",
    "tweet-2lr-1e", "tweet-2lr-3e", "tweet-2lr-5e", "tweet-2lr-10e",
    "tweet-3lr-1e", "tweet-3lr-3e", "tweet-3lr-5e", "tweet-3lr-10e"
]

# F1 Score data for each dataset
f1_data = {
    "labelled1": [0.734, 0.763, 0.741, 0.738, 0.738, 0.713, 0.759, 0.726, 0.753, 0.719, 0.732, 0.795],
    "labelled2": [0.718, 0.741, 0.729, 0.714, 0.707, 0.705, 0.699, 0.706, 0.739, 0.711, 0.705, 0.713]
}

# Calculate the average F1 score for each dataset
avg_f1_labelled1 = np.mean(f1_data["labelled1"])
avg_f1_labelled2 = np.mean(f1_data["labelled2"])

# Plotting
x = np.arange(len(models))  # Model indices
width = 0.35  # Bar width

fig, ax = plt.subplots(figsize=(12, 8))

# Bar charts for F1 scores with adjusted blue and redder orange
ax.bar(x - width / 2, f1_data["labelled1"], width, color='#246B8B', label="F1 Score (labelled1)")  # Blended blue
ax.bar(x + width / 2, f1_data["labelled2"], width, color='#F17D48', label="F1 Score (labelled2)")  # Slightly redder orange

# Plot average lines
ax.axhline(avg_f1_labelled1, color='#246B8B', linestyle='--', label=f"Average (labelled1): {avg_f1_labelled1:.3f}", linewidth=3)
ax.axhline(avg_f1_labelled2, color='#F17D48', linestyle='--', label=f"Average (labelled2): {avg_f1_labelled2:.3f}", linewidth=3)


# Labels, title, and y-axis range
ax.set_xlabel("Model")
ax.set_ylabel("F1 Score")
ax.set_title("F1 Scores of BERTweet Models Fine-Tuned on the Disaster Tweets Dataset")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45)

ax.set_ylim(0.6, 0.8)  # Set y-axis range from 0.6 to 1.0
ax.legend()

# Save and show the plot
plt.tight_layout()
plt.savefig("graphMakers/tweetF1.png")
plt.show()
