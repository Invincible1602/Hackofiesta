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

# Set the similarity threshold to 1.0
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
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

def generate_prompt_template(user_query, word_limit):
    prompt = (
        f"You are an AI assistant knowledgeable in various topics. "
        f"The user has asked: '{user_query}'. "
        f"Please provide a clear, concise, and informative answer in exactly {word_limit} words that fully addresses the query. "
        "If further clarification is needed, feel free to indicate so."
    )
    return prompt


def format_response_template(generated_text):
    response = (
        f"Here is the information you requested:\n\n{generated_text}\n\n"
        "If you have any additional questions, please let me know!"
    )
    return response


def get_gemini_response(user_query, word_limit=100):
    prompt = generate_prompt_template(user_query, word_limit)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(GEMINI_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        return (
            "I'm sorry, but I'm having trouble generating a response right now. Please try again later."
        )
    result = response.json()
    generated_text = (
        result.get("candidates", [{}])[0]
              .get("content", {})
              .get("parts", [{}])[0]
              .get("text", "")
    )
    return format_response_template(generated_text)


def is_medical_query(query: str) -> bool:
    medical_keywords = [
        "medicine", "health", "doctor", "medical", "treatment", "diagnosis", "symptom", "illness", "disease","cure"
    "infection", "surgery", "medication", "therapy", "emergency", "clinic", "hospital", "cataract", "cancer",
    "diabetes", "ulcer", "heart", "cardiac", "kidney", "renal", "liver", "stomach", "headache", "migraine",
    "fever", "flu", "asthma", "allergy", "hypertension", "arthritis", "anxiety", "depression", "stroke",
    "pneumonia", "bronchitis", "covid", "covid-19", "chronic", "acute", "obesity", "pain", "leg pain", "dengue", "malaria", "chickenpox", "chicken pox", "measles", "mumps",
    "arm pain", "back pain", "chest pain", "toothache", "earache", "sore throat", "dizziness", "nausea",
    "vomiting", "diarrhea", "constipation", "rash", "itching", "swelling", "inflammation", "fracture",
    "injury", "trauma", "seizure", "convulsion", "insomnia", "sleep", "appetite", "thyroid", "hormone",
    "sugar", "insulin", "wound", "burn", "sprain", "heartburn", "acid reflux", "indigestion", "bloating",
    "shortness of breath", "palpitations", "cough", "ear infection", "vision", "hearing", "otitis", "eczema",
    "psoriasis", "lupus", "autoimmune", "mental health", "tuberculosis", "strep throat", "sinusitis",
    "pertussis", "whooping cough", "HIV", "AIDS", "hepatitis A", "hepatitis B", "hepatitis C", "malaria",
    "dengue", "Zika", "Lyme disease", "Giardiasis", "salmonellosis", "babesiosis", "chickenpox", "shingles",
    "conjunctivitis", "cold sore", "acne", "dermatitis", "otitis media", "tonsillitis", "oral thrush",
    "osteoporosis", "gout", "bursitis", "hemorrhoids", "Alzheimer's disease", "Parkinson's disease",
    "dementia", "epilepsy", "osteopenia", "anemia", "sickle cell anemia", "hemophilia", "schizophrenia",
    "bipolar disorder", "coronary artery disease", "myocardial infarction", "heart failure", "arrhythmia",
    "cardiomyopathy", "endocarditis", "myocarditis", "aortic aneurysm", "COPD", "emphysema",
    "chronic bronchitis", "pulmonary fibrosis", "bronchiectasis", "pleurisy", "lung cancer", "GERD",
    "peptic ulcer", "Crohn's disease", "ulcerative colitis", "hepatitis", "cirrhosis", "pancreatitis",
    "gallstones", "celiac disease", "diverticulitis", "multiple sclerosis", "Guillain–Barré syndrome",
    "peripheral neuropathy", "meningitis", "encephalitis", "osteoarthritis", "rheumatoid arthritis",
    "muscular dystrophy", "hypothyroidism", "hyperthyroidism", "Cushing's syndrome", "Addison's disease",
    "acromegaly", "urinary tract infection", "nephrolithiasis", "acute kidney injury",
    "chronic kidney disease", "glomerulonephritis", "polycystic kidney disease", "endometriosis",
    "uterine fibroids", "ovarian cyst", "pelvic inflammatory disease", "prostate cancer",
    "erectile dysfunction", "basal cell carcinoma", "cellulitis", "systemic lupus erythematosus",
    "lymphoma", "leukemia", "anaphylaxis", "iron-deficiency anemia", "pernicious anemia",
    "thrombocytopenia", "multiple myeloma", "glaucoma", "age-related macular degeneration",
    "tinnitus", "hearing loss", "measles", "mumps", "rubella", "norovirus", "RSV",
    "Down syndrome", "cystic fibrosis", "Marfan syndrome", "spina bifida", "breast cancer",
    "colorectal cancer", "leukemia", "lymphoma", "melanoma",
    "proteinuria", "hematuria", "dysuria", "polyuria", "oliguria", "nocturia", "urinary retention",
    "urinary incontinence", "interstitial cystitis", "cystitis", "prostatitis", "bladder cancer","urinary"
    "uremia", "creatinine", "BUN", "glomerular filtration rate", "urine", "urinary problem", "urinary tract",
    "fatigue", "exhaustion", "muscle ache", "body ache", "night sweats",
    "chills", "shivering", "cold sweats",
    "runny nose", "stuffy nose", "nasal congestion", "sneezing", "sniffling",
    "coughing", "wheezing",
    "queasy", "nauseous", 
    "pressure", "tightness", "squeezing", "lightheadedness", "feeling faint",
    "pins and needles", "stiff neck", "yawning", "food cravings", "blurred vision",
    "loss of smell", "loss of taste",
    "frequent urination", "thirsty", "weight loss", "weight gain", "cuts heal slowly",
    "difficulty breathing", "shortness of breath", "can't catch my breath", "air hunger", 
    "labored breathing", "shallow breathing", "rapid breathing", "heavy breathing", 
    "gasping for air", "panting", "feeling suffocated", "like I'm drowning", 
    "chest feels tight", "chest feels heavy", "hyperventilating", "noisy breathing", "short breathe"
    "erectile dysfunction", "impotence", "premature ejaculation", "delayed ejaculation", "retrograde ejaculation", "piles",
      "cold", "common cold", "heart attack", "broken bone", "bruise",
    "nosebleed", "food poisoning", "stomach bug", "gastroenteritis",
    "dehydration", "hangover", "sleep apnea", "vertigo", "numbness",
    "high blood pressure", "low blood pressure", "hypotension",
    "irritable bowel syndrome", "IBS", "appendicitis", "hernia",
    "kidney stones", "gas", "flatulence", "belching", "bloody stool",
    "cavity", "tooth decay", "gingivitis", "gum disease", "pink eye",
    "sinus infection", "staph infection", "yeast infection",
    "sunburn", "hives", "blister", "itchy eyes", "eczema flare-up",
    "panic attack", "phobia", "OCD", "PTSD", "autism", "ADHD",
    "restless leg syndrome", "narcolepsy", "binge eating disorder",
    "thyroid disorder", "Hashimoto’s", "Graves’ disease", "PCOS",
    "gestational diabetes", "metabolic syndrome", "vitamin D deficiency",
    "pregnancy", "morning sickness", "miscarriage", "menstrual cramps",
    "PMS", "amenorrhea", "dysmenorrhea", "menopause", "fibromyalgia",
    "BPH", "prostate enlargement", "kidney infection", "pyelonephritis",
    "mono", "mononucleosis", "cholera", "typhoid fever", "tuberculosis (TB)",
    "rotavirus", "norovirus", "RSV", "hepatitis D", "hepatitis E",
    "scleroderma", "Sjogren’s syndrome", "psoriatic arthritis",
    "skin cancer", "squamous cell carcinoma", "melanoma", "breast tumor",
    "lung tumor", "prostate tumor", "pancreatic cancer", "leukemia subtype",
    "lymphoma subtype", "Hodgkin’s lymphoma", "non-Hodgkin’s lymphoma",
    "whooping cough", "pertussis", "RSV", "mumps", "rubella", "chickenpox",
    "down syndrome", "autism spectrum disorder", "spina bifida",
    "PTSD", "addiction", "withdrawal", "electrolyte imbalance",
    "muscle strain", "sciatica", "plantar fasciitis"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in medical_keywords)


def is_conversational_query(query: str) -> bool:
    """
    Detect simple conversational queries like greetings and thanks.
    """
    conv_keywords = [
        "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you"
    ]
    q = query.lower().strip()
    # Check if the user query matches or contains common conversational phrases
    return any(q == kw or q.startswith(kw) or kw in q for kw in conv_keywords)


def get_faq_response(user_query, word_limit=100):
    # Exit case
    if user_query.lower().strip() == "exit":
        return "Thank you for using the FAQ Bot. Have a great day!"

    # Handle conversational queries via Gemini for natural replies
    if is_conversational_query(user_query):
        return get_gemini_response(user_query, word_limit)

    # Embed and search.
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, 1)

    # If no close FAQ match
    if distances[0][0] > SIMILARITY_THRESHOLD:
        if is_medical_query(user_query):
            return get_gemini_response(user_query, word_limit)
        else:
            return (
                "Hmm, I couldn’t find a clear answer to your question. "
                "Could you please try rephrasing it? "
                "If you have any medical problem, book an appointment, or check our website for more details!"
            )

    # Return matched FAQ
    return df.iloc[indices[0][0]]['answer']

@app.get("/faq/")
def faq_query(
    query: str,
    word_limit: int = Query(
        100,
        description="The answer will be provided in exactly this many words (for Gemini responses)"
    )
):
    response = get_faq_response(query, word_limit)
    return {"question": query, "answer": response}

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
