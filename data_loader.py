from datasets import load_dataset

# Download Yelp Polarity dataset automatically
dataset = load_dataset("yelp_polarity")

print("Dataset downloaded successfully!")
print(dataset)
