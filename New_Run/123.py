import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the confusion matrix data
confusion_matrix_data = [
    [94.18, 0.00, 0.00, 0.00, 4.65, 1.17, 0.00],  # Happy
    [1.69, 94.92, 0.00, 0.00, 0.00, 3.39, 0.00],  # Sad
    [0.00, 0.00, 96.34, 0.00, 2.24, 1.22, 0.00],  # Surprise
    [2.99, 2.99, 0.00, 92.54, 0.00, 1.48, 0.00],  # Neutral
    [0.00, 0.00, 1.45, 0.00, 98.55, 0.00, 0.00],  # Fear
    [3.13, 0.00, 0.00, 0.00, 0.00, 95.31, 1.56],  # Disgust
    [0.00, 1.38, 0.00, 0.00, 0.00, 1.38, 97.22]   # Anger
]

# Define emotion labels
emotions = ['Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust', 'Anger']

# Convert the data into a DataFrame for better handling
confusion_df = pd.DataFrame(confusion_matrix_data, index=emotions, columns=emotions)

# Custom formatting function to add '%' symbol
def add_percentage(values):
    return np.array([[f"{val:.2f}%" for val in row] for row in values])

# Get the formatted values
formatted_values = add_percentage(confusion_df.values)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_df, annot=formatted_values, fmt='', cmap='coolwarm', cbar=True, linecolor='darkblue', linewidths=0.5)

# Add labels and title
plt.title("Confusion Matrix Percentages for Emotion Detection", fontsize=16)
plt.xlabel("Predicted Emotion", fontsize=15)
plt.ylabel("True Emotion", fontsize=15)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()