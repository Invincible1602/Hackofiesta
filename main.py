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
import gdown  

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


nltk.download('stopwords')


file_id_embeddings= "1DBPhxW4lqETklPZz7H0ljEkGRJH0bojQ"
file_id_metadata = "1RkJNhOzxVkdY17UAckRbAqSO_XlEU5Ah"


gdown.download(f"https://drive.google.com/uc?export=download&id={file_id_metadata}", "all_metadata.csv", quiet=False)
gdown.download(f"https://drive.google.com/uc?export=download&id={file_id_embeddings}", "all_tfidf_embeddings.csv", quiet=False)


df = pd.read_csv('all_metadata.csv')  
embeddings_df = pd.read_csv('all_tfidf_embeddings.csv') 
model = joblib.load('random_forest_model.pkl') 
vectorizer = joblib.load('vectorizer.pkl') 
label_encoder = joblib.load('label_encoder.pkl') 


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002702-\U000027B0"  
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def correct_text(text: str) -> str:
    blob = TextBlob(text)
    corrected_text = blob.correct()
    return str(corrected_text)


class UserInput(BaseModel):
    comment: str

@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI app!"}


@app.post("/predict/")
async def predict_specialist(user_input: UserInput):
    
    input_text = user_input.comment
    input_text = remove_emojis(input_text)
    input_text = correct_text(input_text)
    
    
    lower_text = input_text.lower()
    
    
    if any(keyword in lower_text for keyword in ["teeth", "gum", "mouth"]):
        final_specialist = "Dentist"
        confidence_score = 100.0
    elif any(keyword in lower_text for keyword in ["throat", "ear", "nose"]):
        final_specialist = "ENT Specialist"
        confidence_score = 100.0
    elif any(keyword in lower_text for keyword in ["backpain", "joints", "bones"]):
        final_specialist = "Orthopedic"
        confidence_score = 100.0
    elif "chest pain" in lower_text:
        final_specialist = "Cardiologist"
        confidence_score = 100.0
    else:
        
        input_tfidf = vectorizer.transform([input_text])
    
        
        prediction = model.predict(input_tfidf)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        
        confidence_score = model.predict_proba(input_tfidf).max() * 100  
    
       
        final_specialist = predicted_label if confidence_score > 70 else "General Physician"
    
    # Return the result as JSON
    return {
        "predicted_specialist": final_specialist,
        "confidence_score": round(confidence_score, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
