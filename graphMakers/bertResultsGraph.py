# import matplotlib.pyplot as plt
# import numpy as np

# # Data for models
# models = [
#     "model-1lr-1e", "model-1lr-3e", "model-1lr-5e", "model-1lr-10e",
#     "model-2lr-1e", "model-2lr-3e", "model-2lr-5e", "model-2lr-10e",
#     "model-3lr-1e", "model-3lr-3e", "model-3lr-5e", "model-3lr-10e"
# ]

# # F1 Score data for each dataset
# f1_data = {
#     "labelled1": [0.821, 0.771, 0.800, 0.840, 0.800, 0.795, 0.790, 0.765, 0.778, 0.825, 0.790, 0.756],
#     "labelled2": [0.642, 0.721, 0.721, 0.690, 0.699, 0.723, 0.706, 0.675, 0.700, 0.729, 0.713, 0.707]
# }

# # Plotting
# x = np.arange(len(models))  # Model indices
# width = 0.35  # Bar width

# fig, ax = plt.subplots(figsize=(12, 8))

# # Bar charts for F1 scores
# ax.bar(x - width / 2, f1_data["labelled1"], width, label="F1 Score (labelled1)")
# ax.bar(x + width / 2, f1_data["labelled2"], width, label="F1 Score (labelled2)")

# # Labels, title, and y-axis range
# ax.set_xlabel("Models")
# ax.set_ylabel("F1 Score")
# ax.set_title("Model F1 Scores by Dataset")
# ax.set_xticks(x)
# ax.set_xticklabels(models, rotation=45)
# ax.set_ylim(0.6, 1.0)  # Set y-axis range from 0.6 to 1.0
# ax.legend()

# # Save and show the plot
# plt.tight_layout()
# plt.savefig("graphMakers/model_f1_scores.png")
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data for models ordered by epochs
models = [
    "model-1lr-1e", "model-2lr-1e", "model-3lr-1e",
    "model-1lr-3e", "model-2lr-3e", "model-3lr-3e",
    "model-1lr-5e", "model-2lr-5e", "model-3lr-5e",
    "model-1lr-10e", "model-2lr-10e", "model-3lr-10e"
]

# F1 Score data reordered by epochs
f1_data = {
    "labelled1": [0.821, 0.800, 0.778, 0.771, 0.795, 0.825, 0.800, 0.790, 0.790, 0.840, 0.765, 0.756],
    "labelled2": [0.642, 0.699, 0.700, 0.721, 0.723, 0.729, 0.721, 0.706, 0.713, 0.690, 0.675, 0.707]
}

# Plotting
x = np.arange(len(models))  # Model indices
width = 0.35  # Bar width

fig, ax = plt.subplots(figsize=(12, 8))

# Bar charts for F1 scores
ax.bar(x - width / 2, f1_data["labelled1"], width, label="F1 Score (labelled1)")
ax.bar(x + width / 2, f1_data["labelled2"], width, label="F1 Score (labelled2)")

# Labels, title, and y-axis range
ax.set_xlabel("Models (Grouped by Epochs)")
ax.set_ylabel("F1 Score")
ax.set_title("Model F1 Scores by Dataset (Grouped by Epochs)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45)
ax.set_ylim(0.6, 0.9)  # Set y-axis range from 0.6 to 1.0
ax.legend()

# Save and show the plot
plt.tight_layout()
plt.savefig("graphMakers/model_f1_scores_by_epochs.png")
plt.show()
