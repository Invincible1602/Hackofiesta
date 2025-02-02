import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import language_tool_python
import joblib

# Ensure you have the NLTK resources
nltk.download('stopwords')

# Load the pre-trained models and data
df = pd.read_csv('all_metadata.csv')  # Load metadata (contains doctor specialist info)
embeddings_df = pd.read_csv('all_tfidf_embeddings.csv')  # Load TF-IDF embeddings
model = joblib.load('random_forest_model.pkl')  # Load the trained Random Forest model
vectorizer = joblib.load('vectorizer.pkl')  # Load the TfidfVectorizer
label_encoder = joblib.load('label_encoder.pkl')  # Load LabelEncoder

tool = language_tool_python.LanguageTool('en-US')
port_stem = PorterStemmer()

# Function to remove emojis from text
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # other symbols
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# Function to stem the text
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Function to correct text (spelling and grammar)
def correct_text(text):
    blob = TextBlob(text)
    corrected_text = blob.correct()
    matches = tool.check(str(corrected_text))
    final_text = language_tool_python.utils.correct(str(corrected_text), matches)
    return final_text

# Streamlit UI
st.title('Patient Doctor Specialist Prediction')
st.write("This app predicts the specialist based on the patient's comments.")

# User Input
user_input = st.text_area("Enter the Patient's Comment")

if user_input:
    # Preprocessing
    user_input = remove_emojis(user_input)
    user_input = stemming(user_input)
    user_input = correct_text(user_input)
    
    # Transform the input using the loaded vectorizer
    input_tfidf = vectorizer.transform([user_input])

    # Prediction using the RandomForest model
    prediction = model.predict(input_tfidf)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    # Show predicted specialist
    st.write(f"Predicted Specialist: {predicted_label}")

    # Optionally, show the model's confidence score
    confidence_score = model.predict_proba(input_tfidf).max()
    st.write(f"Confidence Score: {confidence_score:.2f}")
