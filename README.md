# Cinemoods 🎥🍿

## Project Goal 🎯
The goal of this project is to create a personalized movie recommendation app that leverages emotion recognition to suggest films based on the user's current emotional state. By analyzing movie descriptions using the **BERT-based emotion recognition model**, the app aims to enhance the Netflix viewing experience by offering tailored recommendations that resonate with users' feelings.

## Features ✨
- **Emotion-Based Recommendations** 🎭: Suggests movies that resonate with the user's current emotional state, such as joy, sadness, or fear.

- **Data Cleaning & Preprocessing** 🧹: Ensures movie descriptions are clean and ready for analysis by removing special characters and stopwords.

- **Advanced Emotion Recognition** 🤖: Utilizes the BERT model for accurate emotion predictions based on movie descriptions.

- **User-Friendly Interface** 💻: Provides an interactive and easy-to-navigate web interface for seamless user experience.

- **Diverse Movie Selection** 🎬: Offers a wide range of movie recommendations tailored to various emotions, enhancing viewing options.

- **Easy Integration with Netflix** 📺: Designed to work with Netflix titles, making it convenient for users to find suitable movies on the platform.

## Project Structure 📁
```plaintext
├── Cinemoods
|   ├── Code/
|   ├── model.py                                 # Emotion prediction model
|   ├── UI/                                  
│   ├── app.py                                   # Streamlit UI application
│   ├── Datasets/
│   │   ├── netflix_titles.csv                   # Original dataset
│   │   ├── netflix_cd.csv                       # Cleaned dataset
│   │   └── netflix_titles_with_predictions.csv  # Dataset with predicted emotions                            
│   └── requirements.txt                         # Required packages for the project
├── LICENSE                                      # Licensing information
└── README.md                                    # Project documentation and instructions
```
## Video Output 🎥
Watch the project demo here: 

https://github.com/user-attachments/assets/51b5bfdf-e95b-446b-86fb-b5b7b54d38af

## Requirements 📦
To run this project, ensure you have the following dependencies installed:

- `pandas==1.5.3`
- `nltk==3.6.3`
- `torch==1.13.1`
- `transformers==4.30.0`
- `huggingface-hub==0.16.4`
- `streamlit==1.21.0`
- `matplotlib==3.6.2`

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```
## Usage 🚀
Clone the repository:
```bash
git clone https://github.com/yourusername/SwiftServe.git
```
Navigate to the project directory:
```bash
cd Cinemoods
```
Install the required packages:
```bash
pip install -r requirements.txt
```
Prepare the dataset: 
```bash
# Dataset containing Netflix titles and descriptions
titles_file = "netflix_titles.csv"

# Cleaned dataset with descriptions
cleaned_titles_file = "netflix_cd.csv"

# Dataset with predicted emotions
predicted_emotions_file = "netflix_titles_with_predictions.csv"
```
Run the Streamlit app:
```bash
streamlit run app.py
```
