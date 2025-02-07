import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

app = FastAPI()

# Load FAQ data
def load_faq_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ["question", "answer"]
    return df

file_path = "improved_faq.csv"
df = load_faq_data(file_path)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
faq_embeddings = model.encode(df["question"], convert_to_numpy=True)

# Create FAISS index
d, index = faq_embeddings.shape[1], faiss.IndexFlatL2(faq_embeddings.shape[1])
index.add(faq_embeddings)

# Save FAISS index
faiss.write_index(index, "faq_index.faiss")
df.to_csv("faq_data.csv", index=False)

# Load FAISS index & Data
index = faiss.read_index("faq_index.faiss")
df = pd.read_csv("faq_data.csv")

def get_faq_response(user_query):
    if user_query.lower() == "exit":
        return "Thank you for using the FAQ Bot. Have a great day!"
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, 1)
    return df.iloc[indices[0][0]]['answer']

@app.get("/faq/")
def faq_query(query: str):
    response = get_faq_response(query)
    return {"question": query, "answer": response}

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

