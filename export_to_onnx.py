import torch
from transformers import ElectraForSequenceClassification, AutoTokenizer
import os

# Project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 Load TRAINED MODEL from checkpoint
model_path = os.path.join(BASE_DIR, "results", "checkpoint-6081")

print("Model Path:", model_path)
print("Path Exists:", os.path.exists(model_path))

if not os.path.exists(model_path):
    raise Exception("Checkpoint path not found!")

# ✅ Load fine-tuned model
model = ElectraForSequenceClassification.from_pretrained(model_path)

# ✅ Load ORIGINAL tokenizer (important fix)
tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

model.eval()

# Dummy input
dummy_input = tokenizer(
    "This is a sample review",
    return_tensors="pt",
    max_length=128,
    padding="max_length",
    truncation=True
)

# ONNX output folder
onnx_output_dir = os.path.join(BASE_DIR, "backend", "model", "electra_onnx")
os.makedirs(onnx_output_dir, exist_ok=True)

onnx_model_path = os.path.join(onnx_output_dir, "model.onnx")

# Export
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    onnx_model_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=14,
)

print("✅ ONNX Export Successful!")
print("Saved at:", onnx_model_path)