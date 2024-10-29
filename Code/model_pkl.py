# -*- coding: utf-8 -*-
"""model_pkl.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tjslXP3UWpV3dIh1xEQQYsK5NO2QPCae
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib  # Import joblib for saving the model

nltk.download('stopwords')

df = pd.read_csv("/content/netflix_titles.csv")

print("Dataset Info:")
df.info()

print("\nFirst few rows of the dataset:")
print(df.head())

# Check for missing values in each column
print("\nMissing values in each column:")
print(df.isnull().sum())

# Fill missing values for columns with high numbers of NaNs
df['director'].fillna("Unknown", inplace=True)
df['cast'].fillna("Unknown", inplace=True)
df['country'].fillna("Unknown", inplace=True)

# For columns with few missing values, drop rows
df.dropna(subset=['date_added', 'rating', 'duration'], inplace=True)

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

# Confirm that there are no more missing values
print("\nMissing values after handling:")
print(df.isnull().sum())

# Summary statistics for description length (useful for understanding text length)
df['description_length'] = df['description'].apply(len)
print("\nDescription Length Statistics:")
print(df['description_length'].describe())

# Data Cleaning Function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning function to the 'description' column
df['cleaned_description'] = df['description'].apply(clean_text)

# Remove stop words
stop_words = set(stopwords.words('english'))
df['cleaned_description'] = df['cleaned_description'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# Display cleaned data
print("\nCleaned Descriptions (first few rows):")
print(df[['description', 'cleaned_description']].head())

# Save the cleaned data to a new CSV
df.to_csv("/content/netflix_cd.csv", index=False)

# Load the GoEmotions model and tokenizer
model_name = "bhadresh-savani/bert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set device to CPU
device = 'cpu'
model.to(device)

# Function to predict emotions for a batch of descriptions
def predict_emotions_batch(descriptions, batch_size=16):
    predictions = []

    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]

        # Tokenize the input descriptions
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted class indices for all descriptions
        predicted_class_indices = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(predicted_class_indices)

    return predictions

# Prepare a list of cleaned descriptions from the DataFrame
cleaned_descriptions = df['cleaned_description'].tolist()

# Make predictions for the entire dataset
predicted_emotions = predict_emotions_batch(cleaned_descriptions)

# Add the predictions to the DataFrame
df['predicted_emotion'] = predicted_emotions

# Save the model to a file
joblib.dump(model, 'model.pkl')  # Save the model to a .pkl file

# Optionally, save the updated DataFrame with predictions to a new CSV
df.to_csv("/content/netflix_titles_with_predictions.csv", index=False)

# Display the DataFrame with the new predictions column
print("\nDataFrame with Predicted Emotions:")
print(df[['description', 'cleaned_description', 'predicted_emotion']].head())