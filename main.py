from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import language_tool_python
import joblib
import uvicorn
import gdown  # Import gdown to download files

# Ensure you have the NLTK resources
nltk.download('stopwords')

# Google Drive file link (direct download)
file_id_embeddings= "1DBPhxW4lqETklPZz7H0ljEkGRJH0bojQ"
file_id_metadata = "1RkJNhOzxVkdY17UAckRbAqSO_XlEU5Ah"

# Download the files from Google Drive
gdown.download(f"https://drive.google.com/uc?export=download&id={file_id_metadata}", "all_metadata.csv", quiet=False)
gdown.download(f"https://drive.google.com/uc?export=download&id={file_id_embeddings}", "all_tfidf_embeddings.csv", quiet=False)

# Load the pre-trained models and data
df = pd.read_csv('all_metadata.csv')  # Load metadata (contains doctor specialist info)
embeddings_df = pd.read_csv('all_tfidf_embeddings.csv')  # Load TF-IDF embeddings
model = joblib.load('random_forest_model.pkl')  # Load the trained Random Forest model
vectorizer = joblib.load('vectorizer.pkl')  # Load the TfidfVectorizer
label_encoder = joblib.load('label_encoder.pkl')  # Load LabelEncoder

tool = language_tool_python.LanguageTool('en-US')
port_stem = PorterStemmer()

# Function to remove emojis from text
def remove_emojis(text: str) -> str:
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
def stemming(content: str) -> str:
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Function to correct text (spelling and grammar)
def correct_text(text: str) -> str:
    blob = TextBlob(text)
    corrected_text = blob.correct()
    matches = tool.check(str(corrected_text))
    final_text = language_tool_python.utils.correct(str(corrected_text), matches)
    return final_text

# Define FastAPI app
app = FastAPI()

# Define the input data model
class UserInput(BaseModel):
    comment: str

@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI app!"}

# Define the API endpoint for prediction
@app.post("/predict/")
async def predict_specialist(user_input: UserInput):
    # Preprocessing
    input_text = user_input.comment
    input_text = remove_emojis(input_text)
    input_text = stemming(input_text)
    input_text = correct_text(input_text)
    
    # Transform the input using the loaded vectorizer
    input_tfidf = vectorizer.transform([input_text])

    # Prediction using the RandomForest model
    prediction = model.predict(input_tfidf)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    # Confidence score
    confidence_score = model.predict_proba(input_tfidf).max() * 100  # Convert to percentage

    # Check confidence score threshold
    final_specialist = predicted_label if confidence_score > 70 else "General Physician"

    # Return the result as JSON
    return {
        "predicted_specialist": final_specialist,
        "confidence_score": round(confidence_score, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
