import streamlit as st
import pandas as pd
import pickle
import os

# Emotion label mapping
emotion_labels = {
    0: 'anger',
    1: 'joy',
    2: 'sadness',
    3: 'fear',
    4: 'surprise',
    5: 'disgust'
}

# Reverse the emotion labels dictionary to get emotion name to ID mapping
emotion_to_id = {v: k for k, v in emotion_labels.items()}

model_path = r"E:\Cinemoods\model.pkl"
data_path = r"E:\Cinemoods\Datasets\netflix_titles_with_predictions.csv"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

df = pd.read_csv(data_path)

def recommend_by_emotion(emotion_name, df):
    emotion_id = emotion_to_id[emotion_name]  # Get the emotion ID
    recommendations = df[df['predicted_emotion'] == emotion_id]  # Filter by ID
    return recommendations[['title', 'description']]  # Return all recommendations

st.title("Cinemoods 🎥 🍿")

try:
    st.image(r"E:\Cinemoods\Image\img.png", width=400) 
except Exception as e:
    st.error(f"Error loading image: {e}")

selected_emotion = st.selectbox("Choose an emotion:", list(emotion_labels.values()))

if st.button("Get Recommendations"):
    recommendations = recommend_by_emotion(selected_emotion, df)
    if not recommendations.empty:
        for idx, row in recommendations.iterrows():
            st.write(f"**Title:** {row['title']}")
            st.write(f"**Description:** {row['description']}")
            st.write("---")
    else:
        st.write("No recommendations available for the selected emotion.")
