import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

def load_faq_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ["question", "answer"]
    return df

file_path = "Hackofiesta/chatbot/improved_faq-1.csv"
df = load_faq_data(file_path)

model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = model.encode(df["question"], convert_to_numpy=True)

d = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(faq_embeddings)

faiss.write_index(index, "faq_index.faiss")
df.to_csv("faq_data.csv", index=False)

index = faiss.read_index("faq_index.faiss")
df = pd.read_csv("faq_data.csv")

SIMILARITY_THRESHOLD = 1.0  

def get_faq_response(user_query):
    # Expanded list of disease-related keywords to cover a broader range of health issues
    disease_keywords = [
        "fever", "cough", "flu", "sick", "ill", "infection", "headache", "nausea", 
        "pain", "cold", "weak", "tired", "fatigue", "dizzy", "vomiting", "diarrhea", 
        "ache", "sore", "rash", "swelling", "injury", "bleeding", "burn", "itch", 
        "allergy", "disease", "symptom", "unwell", "hurt", "stomach", "chest", 
        "throat", "breath", "breathing", "temperature", "sweat", "chills", "joint", 
        "muscle", "cramp", "numb", "tingling", "feeling bad", "not feeling well"
    ]
    if any(keyword in user_query.lower() for keyword in disease_keywords):
        disease_responses = [
            "Wishing you a speedy recovery—stay strong! If you need more help, please visit our website.",
            "Sending you positive vibes for a quick recovery! Visit our website if you have any lingering issues.",
            "Hope you feel better soon—take care! Check our website if your symptoms stick around."
        ]
        return random.choice(disease_responses)
    
    if user_query.lower() == "exit":
        return "Thanks for chatting! Take care and have a great day!"

    query_embedding = model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, 1)
    
    if distances[0][0] > SIMILARITY_THRESHOLD:
        return (
            "Hmm, I couldn't find a clear answer to your question. "
            "Could you please try rephrasing it or check our website for more details?"
        )
    
    answer = df.iloc[indices[0][0]]['answer']
    return f"Here's what I found: {answer}"

@app.get("/faq/")
def faq_query(query: str):
    response = get_faq_response(query)
    return {"question": query, "answer": response}

@app.get("/")
def read_root():
    return {"message": "Hello there! How can I help you today?"}

if __name__ == "__main__":
    import uvicorn    
    uvicorn.run(app, host="0.0.0.0", port=8000)