import pandas as pd
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set the similarity threshold to 1.0 as requested.
SIMILARITY_THRESHOLD = 1.0

def load_faq_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ["question", "answer"]
    return df

# Load FAQ data.
file_path = "improved_faq-1.csv"
df = load_faq_data(file_path)

# Load SentenceTransformer model and create FAQ embeddings.
model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = model.encode(df["question"], convert_to_numpy=True)
d = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(faq_embeddings)

# Save index and data.
faiss.write_index(index, "faq_index.faiss")
df.to_csv("faq_data.csv", index=False)

# Reload index and data.
index = faiss.read_index("faq_index.faiss")
df = pd.read_csv("faq_data.csv")

# Gemini API configuration.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def generate_prompt_template(user_query, word_limit):
    """
    Create a detailed prompt template for the Gemini API request,
    instructing the API to generate an answer in exactly {word_limit} words.
    """
    prompt = (
        f"You are an AI assistant knowledgeable in various topics. "
        f"The user has asked: '{user_query}'. "
        f"Please provide a clear, concise, and informative answer in exactly {word_limit} words that fully addresses the query. "
        "If further clarification is needed, feel free to indicate so."
    )
    return prompt

def format_response_template(generated_text):
    """
    Format the generated text into a well-structured response without truncation.
    """
    response = (
        f"Here is the information you requested:\n\n{generated_text}\n\n"
        "If you have any additional questions, please let me know!"
    )
    return response

def get_gemini_response(user_query, word_limit=50):
    """
    Call the Gemini API using a prompt template and handle any errors gracefully.
    """
    prompt = generate_prompt_template(user_query, word_limit)
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(GEMINI_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        try:
            error_details = response.json()
        except Exception:
            error_details = {"error": "Unable to parse error details"}
        print("Gemini API Error:", error_details)
        return ("I'm sorry, but I'm having trouble generating a response right now. Please try again later.")
    
    result = response.json()
    generated_text = (
        result.get("candidates", [{}])[0]
              .get("content", {})
              .get("parts", [{}])[0]
              .get("text", "")
    )
    return format_response_template(generated_text)

def is_medical_query(query: str) -> bool:
    """
    Check if the user's query contains any keywords related to medical diseases.
    """
    medical_keywords = [
        "medicine", "health", "doctor", "medical", "treatment", "diagnosis", "symptom", "illness", "disease",
        "infection", "surgery", "medication", "therapy", "emergency", "clinic", "hospital",
        "cancer", "diabetes", "ulcer", "heart", "cardiac", "kidney", "renal", "liver", "stomach", "headache", 
        "migraine", "fever", "flu", "asthma", "allergy", "hypertension", "arthritis", "anxiety", "depression",
        "stroke", "pneumonia", "bronchitis", "covid", "covid-19", "chronic", "acute", "obesity",
        "pain", "leg pain", "arm pain", "back pain", "chest pain", "toothache", "earache", "sore throat",
        "dizziness", "nausea", "vomiting", "diarrhea", "constipation", "rash", "itching", "swelling", 
        "inflammation", "fracture", "injury", "trauma", "seizure", "convulsion", "insomnia", "sleep", 
        "appetite", "thyroid", "hormone", "sugar", "insulin", "wound", "burn", "sprain", "heartburn",
        "acid reflux", "indigestion", "bloating", "shortness of breath", "palpitations", "cough", "ear infection",
        "vision", "hearing", "otitis", "eczema", "psoriasis", "lupus", "autoimmune", "mental health"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in medical_keywords)

def get_faq_response(user_query, word_limit=50):
    """
    Returns the FAQ answer if a similar question is found in the CSV file.
    If the similarity score is above the threshold, then it checks if the query is medical-related.
    If it is, it generates a response via the Gemini API;
    otherwise, it returns a message asking the user to rephrase the query.
    """
    if user_query.lower().strip() == "exit":
        return "Thank you for using the FAQ Bot. Have a great day!"
    
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, 1)
    
    if distances[0][0] > SIMILARITY_THRESHOLD:
        # If the similarity score is too high (i.e., not similar enough), check for medical keywords.
        if is_medical_query(user_query):
            return get_gemini_response(user_query, word_limit)
        else:
            return (
                "Hmm, I couldn’t find a clear answer to your question. Could you please try rephrasing it? "
                "If you have any medical problem, book an appointment, or check our website for more details!"
            )
    
    return df.iloc[indices[0][0]]['answer']

@app.get("/faq/")
def faq_query(query: str, word_limit: int = Query(50, description="The answer will be provided in exactly this many words (for Gemini responses)")):
    response = get_faq_response(query, word_limit)
    return {"question": query, "answer": response}

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn    
    uvicorn.run(app, host="0.0.0.0", port=8000)
