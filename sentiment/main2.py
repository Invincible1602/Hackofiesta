from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import json
import pickle
import gdown
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import os

app = FastAPI()

# Extracted File IDs from Google Drive links
MODEL_ID = "1p5ZWJ9I1j4yenHJMUt-EpGOqXiZImAtv"
TAGS_ID = "1RjVrY91Zt7tRpb9MplE6fRJ5drSTFlf6"
EMBEDDINGS_ID = "1OGWIhO7p5e7EwhVH3Ve3v5xT2Tpu1NRz"

# Local paths
MODEL_PATH = "save_model.pkl"
TAGS_PATH = "save_tags.json"
EMBEDDINGS_PATH = "save_embeddings.json"

# Function to download files if missing
def download_file(url, path):
    if not os.path.exists(path):
        gdown.download(url, path, quiet=False)

# Download necessary files
download_file(f"https://drive.google.com/uc?export=download&id={MODEL_ID}", MODEL_PATH)
download_file(f"https://drive.google.com/uc?export=download&id={TAGS_ID}", TAGS_PATH)
download_file(f"https://drive.google.com/uc?export=download&id={EMBEDDINGS_ID}", EMBEDDINGS_PATH)

# Lazy loading
model = None
sentiment_pipeline = None
tags = None
embeddings = None

def load_resources():
    global model, sentiment_pipeline, tags, embeddings

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    if sentiment_pipeline is None:
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", trust_remote_code=False)

    if tags is None:
        with open(TAGS_PATH, "r") as f:
            tags = json.load(f)

    if embeddings is None:
        with open(EMBEDDINGS_PATH, "r") as f:
            embeddings = json.load(f)

        # Convert embeddings to PyTorch tensors (float16 for lower memory usage)
        for key in embeddings:
            embeddings[key] = torch.tensor(embeddings[key], dtype=torch.float16, device="cpu")

class FeedbackRequest(BaseModel):
    feedback: str = Field(..., min_length=1, description="Feedback text must not be empty")

def classify_feedback(feedback: str):
    """Classifies feedback as positive, negative, or neutral using embeddings and sentiment analysis."""

    load_resources()  # Load models only when needed

    feedback = feedback.strip()  # Remove leading/trailing spaces
    if not feedback:
        raise HTTPException(status_code=400, detail="Feedback cannot be empty or just spaces")

    feedback_embedding = model.encode(feedback, convert_to_tensor=True, device="cpu").half()

    # Neutral keyword-based classification
    neutral_keywords = {"normal", "basic", "average", "fine", "okay", "decent",
        "standard", "ordinary", "regular", "common", "nothing",
        "usual", "necessary", "general", "typical", "neutral"}
    if any(word in feedback.lower() for word in neutral_keywords):
        best_match_index = torch.argmax(util.pytorch_cos_sim(feedback_embedding, embeddings["neutral"])).item()
        return "neutral", tags["neutral"][best_match_index]

    # Compute similarity for neutral feedback
    neutral_similarity_scores = util.pytorch_cos_sim(feedback_embedding, embeddings["neutral"])
    if torch.max(neutral_similarity_scores).item() > 0.7:
        best_match_index = torch.argmax(neutral_similarity_scores).item()
        return "neutral", tags["neutral"][best_match_index]

    # Sentiment analysis
    sentiment_result = sentiment_pipeline(feedback)[0]
    sentiment_label = sentiment_result["label"]
    sentiment_score = sentiment_result["score"]

    # Classify based on sentiment
    if sentiment_label == "POSITIVE" and sentiment_score >= 0.65:
        tag_category = "positive"
        tag_embeddings = embeddings["positive"]
        tag_list = tags["positive"]
    elif sentiment_label == "NEGATIVE" and sentiment_score >= 0.65:
        tag_category = "negative"
        tag_embeddings = embeddings["negative"]
        tag_list = tags["negative"]
    else:
        tag_category = "neutral"
        tag_embeddings = embeddings["neutral"]
        tag_list = tags["neutral"]

    # Find the best matching tag
    similarity_scores = util.pytorch_cos_sim(feedback_embedding, tag_embeddings)
    best_match_index = torch.argmax(similarity_scores).item()
    best_tag = tag_list[best_match_index]

    return tag_category, best_tag

@app.post("/classify/")
async def classify(request: FeedbackRequest):
    sentiment, tag = classify_feedback(request.feedback)
    return {"sentiment": sentiment, "tag": tag}

@app.get("/")
async def home():
    return {"message": "Doctor Feedback Sentiment Analysis API is running!"}

if __name__ == "__main__":
    import uvicorn    
    uvicorn.run(app, host="0.0.0.0", port=8000)
