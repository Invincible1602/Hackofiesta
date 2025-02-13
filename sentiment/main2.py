from fastapi import FastAPI
from pydantic import BaseModel
import torch
import json
import pickle
import gdown
from sentence_transformers import util
from transformers import pipeline

app = FastAPI()

# Extracted File IDs from Google Drive links
MODEL_ID = "1p5ZWJ9I1j4yenHJMUt-EpGOqXiZImAtv"
TAGS_ID = "1RjVrY91Zt7tRpb9MplE6fRJ5drSTFlf6"
EMBEDDINGS_ID = "1OGWIhO7p5e7EwhVH3Ve3v5xT2Tpu1NRz"

# Construct direct download URLs
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"
TAGS_URL = f"https://drive.google.com/uc?export=download&id={TAGS_ID}"
EMBEDDINGS_URL = f"https://drive.google.com/uc?export=download&id={EMBEDDINGS_ID}"

# Local paths
MODEL_PATH = "save_model.pkl"
TAGS_PATH = "save_tags.json"
EMBEDDINGS_PATH = "save_embeddings.json"

# Download files
gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
gdown.download(TAGS_URL, TAGS_PATH, quiet=False)
gdown.download(EMBEDDINGS_URL, EMBEDDINGS_PATH, quiet=False)

# Load the saved model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load saved tags and embeddings
with open(TAGS_PATH, "r") as f:
    tags = json.load(f)

with open(EMBEDDINGS_PATH, "r") as f:
    embeddings = json.load(f)

# Convert embeddings to tensors
positive_embeddings = torch.tensor(embeddings["positive"])
negative_embeddings = torch.tensor(embeddings["negative"])
neutral_embeddings = torch.tensor(embeddings["neutral"])

class FeedbackRequest(BaseModel):
    feedback: str

def classify_feedback(feedback):
    """Classifies feedback as positive, negative, or neutral based on keywords and embeddings."""

    # Neutral keyword-based classification
    neutral_keywords = {
        "normal", "basic", "average", "fine", "okay", "decent",
        "standard", "ordinary", "regular", "common", "nothing",
        "usual", "necessary", "general", "typical", "neutral"
    }

    if any(word in feedback.lower() for word in neutral_keywords):
        best_match_index = torch.argmax(util.pytorch_cos_sim(
            model.encode(feedback, convert_to_tensor=True), neutral_embeddings
        )).item()
        return "neutral", tags["neutral"][best_match_index]

    # Proceed with embedding similarity for neutral classification
    feedback_embedding = model.encode(feedback, convert_to_tensor=True)
    neutral_similarity_scores = util.pytorch_cos_sim(feedback_embedding, neutral_embeddings)

    best_neutral_score = torch.max(neutral_similarity_scores).item()
    if best_neutral_score > 0.7:  # Increased threshold to make neutral more likely
        best_match_index = torch.argmax(neutral_similarity_scores).item()
        return "neutral", tags["neutral"][best_match_index]

    # Sentiment analysis
    sentiment_result = sentiment_pipeline(feedback)[0]
    sentiment_label = sentiment_result["label"]
    sentiment_score = sentiment_result["score"]

    # Stricter sentiment thresholds
    if sentiment_label == "POSITIVE" and sentiment_score >= 0.65:
        tag_category = "positive"
        tag_embeddings = positive_embeddings
        tag_list = tags["positive"]
    elif sentiment_label == "NEGATIVE" and sentiment_score >= 0.65:
        tag_category = "negative"
        tag_embeddings = negative_embeddings
        tag_list = tags["negative"]
    else:
        tag_category = "neutral"
        tag_embeddings = neutral_embeddings
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
