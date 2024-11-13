import matplotlib.pyplot as plt
import numpy as np

# Data for models
models = [
    "model-1lr-1e", "model-1lr-3e", "model-1lr-5e", "model-1lr-10e",
    "model-2lr-1e", "model-2lr-3e", "model-2lr-5e", "model-2lr-10e",
    "model-3lr-1e", "model-3lr-3e", "model-3lr-5e", "model-3lr-10e"
]

# F1 Score data for each dataset
f1_data = {
    "labelled1": [0.821, 0.771, 0.800, 0.840, 0.800, 0.795, 0.790, 0.765, 0.778, 0.825, 0.790, 0.756],
    "labelled2": [0.642, 0.721, 0.721, 0.690, 0.699, 0.723, 0.706, 0.675, 0.700, 0.729, 0.713, 0.707]
}

# Calculate the average F1 score for each dataset
avg_f1_labelled1 = np.mean(f1_data["labelled1"])
avg_f1_labelled2 = np.mean(f1_data["labelled2"])

# Plotting
x = np.arange(len(models))  # Model indices
width = 0.35  # Bar width

fig, ax = plt.subplots(figsize=(12, 8))

# Bar charts for F1 scores
ax.bar(x - width / 2, f1_data["labelled1"], width, color='#2A5D78', label="F1 Score (labelled1)")
ax.bar(x + width / 2, f1_data["labelled2"], width, color='#F6AA54', label="F1 Score (labelled2)")

# Plot average lines
ax.axhline(avg_f1_labelled1, color='#2A5D78', linestyle='--', label=f"Average (labelled1): {avg_f1_labelled1:.3f}", linewidth=3)
ax.axhline(avg_f1_labelled2, color='#F6AA54', linestyle='--', label=f"Average (labelled2): {avg_f1_labelled2:.3f}", linewidth=3)

# Labels, title, and y-axis range
ax.set_xlabel("Model")
ax.set_ylabel("F1 Score")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45)

ax.set_ylim(0.6, 0.85)  # Set y-axis range from 0.6 to 1.0
ax.legend()

# Save and show the plot
plt.tight_layout()
plt.savefig("graphMakers/f1scoresforBERT")
plt.show()
