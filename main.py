from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from textblob import TextBlob
import requests
import os
from dotenv import load_dotenv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

class UserInput(BaseModel):
    comment: str

# Text preprocessing functions

def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
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
    return str(blob.correct())

# Generate enhanced prompt for Gemini

def generate_specialist_prompt(symptoms: str) -> str:
    return (
        "You are a medical specialist recommendation assistant.\n"
        "First, determine if the symptoms indicate a physical injury or a mental health issue.\n"
        "Then, provide only the single best specialist for that category without explanation.\n"
        "Examples:\n"
        "- 'I have a red, painful burn on my hand' -> Dermatologist\n"
        "- 'I feel exhausted, detached, and cynical about my job' -> Psychologist\n"
        f"Now, given the symptoms: '{symptoms}', provide the specialist."
    )

# Call to Gemini API
def get_gemini_specialist(symptoms: str) -> str:
    prompt = generate_specialist_prompt(symptoms)
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_ENDPOINT, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    specialist = (
        result.get("candidates", [{}])[0]
              .get("content", {})
              .get("parts", [{}])[0]
              .get("text", "General Physician")
    ).strip().strip('"')
    return specialist or "General Physician"

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Specialist Detection API with Gemini!"}

@app.post("/predict/")
async def predict_specialist(user_input: UserInput):
    raw_text = user_input.comment

    # Pre-process override for burnout on raw input
    if re.search(r'\bburn[- ]?out\b', raw_text, flags=re.IGNORECASE):
        return {"predicted_specialist": "Psychologist", "confidence_score": 100.0}

    # Apply text cleaning and correction
    input_text = remove_emojis(raw_text)
    input_text = correct_text(input_text)

    # Use Gemini for all cases
    try:
        gemini_specialist = get_gemini_specialist(input_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    # Normalize orthopedics-related responses
    if re.search(r'orthoped', gemini_specialist, flags=re.IGNORECASE):
        normalized = "Orthopedic"
    else:
        normalized = gemini_specialist

    return {"predicted_specialist": normalized, "confidence_score": 0.0}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
