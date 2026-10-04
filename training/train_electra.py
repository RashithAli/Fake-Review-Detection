import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("data/final_fake_review_dataset.csv")

df = df.dropna()
df["label"] = df["label"].astype(int)

print("Dataset size:", len(df))
print(df["label"].value_counts())

# ---------------------------
# Train / Test Split
# ---------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})

# ---------------------------
# Tokenizer
# ---------------------------
tokenizer = ElectraTokenizer.from_pretrained(
    "google/electra-small-discriminator"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=96
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ---------------------------
# Model
# ---------------------------
model = ElectraForSequenceClassification.from_pretrained(
    "google/electra-small-discriminator",
    num_labels=2
)

model.to(device)

# ---------------------------
# Metrics
# ---------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ---------------------------
# Training Arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",              # updated (no warning)
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none"
)

# ---------------------------
# Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

# ---------------------------
# Train
# ---------------------------
print("Training started...")
trainer.train()

# ---------------------------
# Save Model (IMPORTANT)
# ---------------------------
save_path = "model/electra_fake_review"
os.makedirs(save_path, exist_ok=True)

trainer.save_model(save_path)  # 🔥 saves pytorch_model.bin
tokenizer.save_pretrained(save_path)

print("✅ Training completed and model saved properly!")