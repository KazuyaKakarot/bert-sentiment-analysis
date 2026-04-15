# Sentiment Analysis API — Fine-tuned DistilBERT

A REST API that classifies movie reviews as Positive or Negative using a fine-tuned DistilBERT model trained on the IMDb dataset.

## Results

| Model | F1 Score | Accuracy |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.8830 | 88.00% |
| Fine-tuned DistilBERT | **0.9089** | **90.72%** |

**Improvement: +2.59 points (+2.9%) over baseline**

## Project Structure
bert-sentiment/
├── baseline.py        # TF-IDF + Logistic Regression baseline
├── train.py           # DistilBERT fine-tuning (run on Google Colab)
├── api.py             # FastAPI inference server
├── requirements.txt   # Dependencies
└── data/
└── final_results.csv

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/bert-sentiment-analysis
cd bert-sentiment-analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

Training was done on Google Colab (T4 GPU) for 3 epochs using 15,000 IMDb reviews.

```bash
# Run on Google Colab
python3 train.py
```

## Run the API

```bash
uvicorn api:app --reload
```

API runs at `http://127.0.0.1:8000`

## API Endpoints

### `GET /health`
```json
{"status": "healthy", "model": "distilbert-base-uncased-finetuned-imdb"}
```

### `POST /predict`

**Request:**
```json
{"text": "This movie was absolutely brilliant!"}
```

**Response:**
```json
{
  "label": "Positive",
  "confidence": 0.9876,
  "scores": {
    "Negative": 0.0124,
    "Positive": 0.9876
  }
}
```

## Tech Stack

- **Model:** DistilBERT (distilbert-base-uncased) fine-tuned on IMDb
- **Framework:** PyTorch + Hugging Face Transformers
- **API:** FastAPI + Uvicorn
- **Training:** Google Colab T4 GPU

## Author
Mohd Ali Haider Zaidi
