from datasets import load_dataset
from transformers import ElectraTokenizer

# Load dataset
dataset = load_dataset("yelp_polarity")

# Load ELECTRA tokenizer
tokenizer = ElectraTokenizer.from_pretrained(
    "google/electra-small-discriminator"
)

# Text cleaning function
def preprocess_text(example):
    text = example["text"].lower().strip()
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["labels"] = example["label"]
    return tokens

# Apply preprocessing
processed_dataset = dataset.map(preprocess_text, batched=False)

# Remove unnecessary columns
processed_dataset = processed_dataset.remove_columns(["text"])

# Set dataset format for PyTorch
processed_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

print("Preprocessing completed successfully!")
print(processed_dataset)
