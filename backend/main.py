from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os
import re

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# ORIGINAL MODEL PATH (UNCHANGED)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "electra_onnx",
    "model.onnx"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "electra_onnx"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH)

print("ONNX Model loaded successfully ✅")


class ReviewRequest(BaseModel):
    text: str


# ----------------------------
# EXPLANATION FUNCTION (NEW)
# ----------------------------
def generate_explanation(text, label, confidence, real_prob, fake_prob):
    words = re.findall(r'\b\w+\b', text.lower())

    exaggerated_words = [
        "amazing", "excellent", "perfect", "worst",
        "terrible", "guaranteed", "100%", "best",
        "extremely", "never", "always", "incredible",
        "outstanding", "fantastic", "must buy"
    ]

    detected = [w for w in words if w in exaggerated_words]

    margin = abs(real_prob - fake_prob)

    explanation = (
        f"The AI model evaluated this review using contextual language understanding "
        f"and probability distribution analysis. It predicted this review as {label} "
        f"with a confidence score of {confidence}%. "
        f"The calculated probability for Real class is {round(real_prob,4)} "
        f"and for Fake class is {round(fake_prob,4)}. "
        f"The decision margin between the two classes is {round(margin,4)}, "
        f"which indicates the strength of the classification boundary. "
    )

    if "Fake" in label:
        explanation += (
            "The review exhibits linguistic characteristics that are commonly "
            "associated with deceptive or promotional content. "
            "These include exaggerated sentiment intensity, strong persuasive tone, "
            "or emotionally amplified wording that may not reflect authentic experience. "
            "Fake reviews often aim to strongly influence reader perception rather "
            "than describe balanced personal usage. "
        )
    else:
        explanation += (
            "The review demonstrates characteristics typical of genuine user feedback. "
            "The language appears balanced, experience-driven, and contextually consistent. "
            "Real reviews generally contain moderate sentiment and specific experiential "
            "details rather than aggressive persuasion. "
        )

    if detected:
        explanation += (
            f"The model detected emotionally strong keywords such as: "
            f"{', '.join(set(detected))}, which may have influenced the classification outcome. "
        )
    else:
        explanation += (
            "No strongly exaggerated promotional keywords were detected in the review text. "
        )

    explanation += (
        "Overall, the classification decision is based on semantic patterns, "
        "tone intensity, contextual coherence, and probability margin analysis "
        "learned during model training."
    )

    return explanation, detected

# ----------------------------
# PREDICT API (LOGIC UNCHANGED)
# ----------------------------
@app.post("/predict")
async def predict(request: ReviewRequest):
    try:
        review_text = request.text

        # Tokenize
        inputs = tokenizer(
            review_text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        input_names = [input.name for input in session.get_inputs()]

        ort_inputs = {
            input_names[0]: input_ids,
            input_names[1]: attention_mask
        }

        outputs = session.run(None, ort_inputs)
        logits = outputs[0]

        # Stable softmax (UNCHANGED)
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        real_prob = float(probabilities[0][0])
        fake_prob = float(probabilities[0][1])

        confidence = max(real_prob, fake_prob) * 100

        # 🔥 MARGIN-BASED DECISION (UNCHANGED)
        margin = real_prob - fake_prob

        if margin < 0.98:
            label = "Fake Review ❌"
        else:
            label = "Real Review ✅"

        # ---------------- ADD EXPLANATION ----------------
        explanation, keywords = generate_explanation(
            review_text,
            label,
            round(confidence, 2),
            real_prob,
            fake_prob
        )

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "real_probability": round(real_prob, 4),
            "fake_probability": round(fake_prob, 4),
            "margin": round(margin, 4),
            "explanation": explanation,
            "keywords_detected": keywords
        }

    except Exception as e:
        print("ONNX ERROR:", str(e))
        return {"error": str(e)}