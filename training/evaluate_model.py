import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import ElectraTokenizer, ElectraForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datasets import Dataset

# ----------------------------
# Load trained model
# ----------------------------
MODEL_PATH = "backend/model/electra_fake_review"

tokenizer = ElectraTokenizer.from_pretrained(MODEL_PATH)
model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ----------------------------
# Load CUSTOM dataset
# ----------------------------
# CSV must contain columns: text, label
df = pd.read_csv("./data/final_fake_review_dataset.csv")

print("Dataset columns:", df.columns)

# Keep only required columns
df = df[["text", "label"]]

# Remove null rows
df.dropna(inplace=True)

# Ensure text is string
df["text"] = df["text"].astype(str)

# Optional: reduce size for faster evaluation
df = df.sample(n=min(3000, len(df)), random_state=42)

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# ----------------------------
# Tokenization (SAFE)
# ----------------------------
def tokenize(example):
    text = example["text"]

    if not isinstance(text, str) or text.strip() == "":
        text = "empty review"

    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=96
    )

dataset = dataset.map(tokenize, batched=False)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ----------------------------
# Prediction
# ----------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for item in dataset:
        inputs = {
            "input_ids": item["input_ids"].unsqueeze(0),
            "attention_mask": item["attention_mask"].unsqueeze(0)
        }

        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

        y_pred.append(pred)
        y_true.append(item["labels"].item())

# ----------------------------
# Evaluation Metrics
# ----------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nEvaluation Results")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# ----------------------------
# Plot Metrics
# ----------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [accuracy, precision, recall, f1]

plt.figure()
plt.bar(metrics, values)
plt.ylim(0, 1)
plt.title("ELECTRA Model Evaluation Metrics")

for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

plt.show()

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "Genuine"],
    yticklabels=["Fake", "Genuine"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
