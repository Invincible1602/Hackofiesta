from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import joblib
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import gdown  # Import gdown to download files

app = FastAPI()

# Add CORS middleware to allow requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

# Function to correct text (spelling)
def correct_text(text: str) -> str:
    blob = TextBlob(text)
    corrected_text = blob.correct()
    return str(corrected_text)

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
