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

# Download metadata and embeddings
file_id_embeddings = "1DBPhxW4lqETklPZz7H0ljEkGRJH0bojQ"
file_id_metadata = "1RkJNhOzxVkdY17UAckRbAqSO_XlEU5Ah"

gdown.download(
    f"https://drive.google.com/uc?export=download&id={file_id_metadata}",
    "all_metadata.csv",
    quiet=False
)
gdown.download(
    f"https://drive.google.com/uc?export=download&id={file_id_embeddings}",
    "all_tfidf_embeddings.csv",
    quiet=False
)

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

# Hardcoded mapping from keywords to specialists
SPECIALIST_MAP = {
    "Cardiologist": {"heart", "chest", "chestpain", "cardio"},
    "Dentist": {"teeth", "gum", "mouth", "toothache", "cavity"},
    "ENT Specialist": {"ear", "nose", "throat", "sinus"},
    "Orthopedic": {"backpain", "spine", "joints", "bones", "arthritis"},
    "Dermatologist": {"skin", "rash", "acne", "eczema", "psoriasis"},
    "Ophthalmologist": {"eye", "vision", "blurry", "glaucoma", "cataract"},
    "Neurologist": {"headache", "migraine", "brain", "seizure", "nervous"},
    "Gastroenterologist": {"stomach", "abdomen", "digestion", "ulcer", "ibs", "colitis"},
    "Pulmonologist": {"lungs", "breath", "asthma", "cough", "pneumonia"},
    "Nephrologist": {"kidney", "renal", "urine", "dialysis"},
    "Urologist": {"urinary", "bladder", "prostate", "incontinence"},
    "Gynecologist": {"pregnancy", "uterus", "ovary", "menstrual", "period"},
    "Psychiatrist": {"mental", "depression", "anxiety", "psycho", "therapy"},
    "Endocrinologist": {"diabetes", "thyroid", "hormone", "metabolism"},
    "Hematologist": {"blood", "anemia", "platelet", "clot"},
    "Rheumatologist": {"autoimmune", "rheumatic", "lupus", "jointpain"},
    "Allergist": {"allergy", "asthma", "hives", "allergic"},
    "Oncologist": {"cancer", "tumor", "chemotherapy", "biopsy"},
   
}

@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI app!"}

@app.post("/predict/")
async def predict_specialist(user_input: UserInput):
    input_text = user_input.comment
    input_text = remove_emojis(input_text)
    input_text = correct_text(input_text)

    lower_text = input_text.lower()
    words = set(re.findall(r"\w+", lower_text))

    # Check each specialist mapping
    for specialist, keywords in SPECIALIST_MAP.items():
        if words.intersection(keywords):
            return {"predicted_specialist": specialist, "confidence_score": 100.0}

    # Fallback to ML model
    input_tfidf = vectorizer.transform([input_text])
    prediction = model.predict(input_tfidf)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    confidence_score = model.predict_proba(input_tfidf).max() * 100
    final_specialist = predicted_label if confidence_score > 70 else "General Physician"

    return {
        "predicted_specialist": final_specialist,
        "confidence_score": round(confidence_score, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
