from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

# ── Load model ─────────────────────────────────────────────────────────
MODEL_PATH = "models/distilbert-final"

print("Loading model...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded and ready!")

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Classifies text as Positive or Negative using fine-tuned DistilBERT",
    version="1.0.0"
)

# ── Request/Response schemas ───────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    label:      str
    confidence: float
    scores:     dict

# ── Routes ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "distilbert-base-uncased-finetuned-imdb"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input: TextInput):
    # Tokenize
    tokens = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    # Inference
    with torch.no_grad():
        outputs = model(**tokens)
        probs   = F.softmax(outputs.logits, dim=1)

    # Parse results
    confidence, predicted_class = torch.max(probs, dim=1)
    label = "Positive" if predicted_class.item() == 1 else "Negative"

    return PredictionOutput(
        label=label,
        confidence=round(confidence.item(), 4),
        scores={
            "Negative": round(probs[0][0].item(), 4),
            "Positive": round(probs[0][1].item(), 4)
        }
    )

@app.post("/predict/batch")
def predict_batch(inputs: list[TextInput]):
    results = []
    for item in inputs:
        results.append(predict(item))
    return results