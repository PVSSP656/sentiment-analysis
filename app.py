import re
from pathlib import Path
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib

# Paths to exported models
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
VEC_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = ARTIFACTS_DIR / "logistic_regression_model.joblib"

# Global state for loaded models
models: Dict = {}


def clean_review_text(text: str) -> str:
    """
    Applies the identical text-cleaning pipeline used in the notebook:
    1. Lowercase
    2. Strip HTML tags (<.*?>)
    3. Strip URLs (https?://... or www...)
    4. Collapse and trim extraneous whitespace
    Preserves all meaningful punctuation and negation tokens.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the exact artifacts trained in Step 7
    if not VEC_PATH.exists() or not MODEL_PATH.exists():
        raise RuntimeError(
            f"Required model artifacts missing in {ARTIFACTS_DIR}. "
            "Ensure Step 10 serialization in notebooks/sentiment_analysis.ipynb has executed."
        )
    models["vectorizer"] = joblib.load(VEC_PATH)
    models["model"] = joblib.load(MODEL_PATH)
    yield
    # Shutdown: clean up
    models.clear()


app = FastAPI(
    title="Amazon Review Sentiment Analysis API",
    description=(
        "Local inference API demonstrating sentiment prediction using a TF-IDF "
        "+ Balanced Logistic Regression model trained on Amazon customer reviews."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ReviewRequest(BaseModel):
    review: str = Field(
        ...,
        description="The customer review text to classify.",
        examples=["The television display is vibrant, but the battery life is poor."],
    )


class PredictionResponse(BaseModel):
    review: str
    predicted_sentiment: str
    confidence: float
    class_probabilities: Dict[str, float]
    note: str = (
        "Confidence is the model's estimated class probability from logistic regression, "
        "not a guarantee of human-verified truth."
    )


@app.get("/", tags=["General"])
async def root():
    """Returns a simple API status message."""
    return {
        "status": "online",
        "service": "Amazon Review Sentiment Analysis API",
        "docs_url": "/docs",
        "description": "POST a review to /predict for sentiment classification.",
    }


@app.get("/health", tags=["General"])
async def health():
    """Returns the operational health and artifact loading status."""
    is_loaded = "vectorizer" in models and "model" in models
    return {
        "status": "healthy" if is_loaded else "degraded",
        "artifacts_loaded": is_loaded,
        "model_type": "Logistic Regression (Balanced)",
        "vectorizer": "TF-IDF (20,000 max features, n-gram 1-2)",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_sentiment(payload: ReviewRequest):
    """
    Accepts customer review text, applies notebook standardisation,
    and returns predicted rating-derived sentiment and probabilities.
    """
    raw_review = payload.review

    # Validate that review is not empty or whitespace-only (HTTP 422)
    if not raw_review or not raw_review.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Review text cannot be empty or contain only whitespace.",
        )

    try:
        # Preprocess text identically to notebook
        cleaned = clean_review_text(raw_review)
        if not cleaned:
            # Review contained only URLs or tags which were stripped out
            cleaned = raw_review.strip().lower()

        # Vectorize
        vectorizer = models["vectorizer"]
        model = models["model"]

        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probabilities = model.predict_proba(vec)[0]

        # Map class probabilities
        classes = model.classes_
        class_probs = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probabilities)}
        confidence = round(float(max(probabilities)), 4)

        return PredictionResponse(
            review=raw_review,
            predicted_sentiment=prediction,
            confidence=confidence,
            class_probabilities=class_probs,
        )

    except HTTPException:
        raise
    except Exception:
        # Generic graceful error handling without exposing internal stack traces
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal error occurred while processing the prediction request."},
        )
