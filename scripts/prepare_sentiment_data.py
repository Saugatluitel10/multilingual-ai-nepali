import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("data/raw/nepali_sentiment.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Define the label mapping
# Dataset: 0 (Negative), 1 (Positive), 2 (Neutral)
# Target Model: 0 (Negative), 1 (Neutral), 2 (Positive)
label_map = {
    0: 0,  # Negative -> Negative
    1: 2,  # Positive -> Positive
    2: 1   # Neutral  -> Neutral
}

print("Remapping labels...")
df['label'] = df['label'].map(label_map)

# Ensure all labels were mapped correctly
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)

print(f"Dataset has {len(df)} samples after cleaning.")
print("Label distribution:")
print(df['label'].value_counts())

# Split the data
print("Splitting data into training and validation sets...")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Create processed directory if it doesn't exist
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# Save the files
train_path = os.path.join(output_dir, "sentiment_train.csv")
val_path = os.path.join(output_dir, "sentiment_val.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)

print(f"Training data saved to {train_path}")
print(f"Validation data saved to {val_path}")
