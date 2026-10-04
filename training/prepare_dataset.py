from datasets import load_dataset
import pandas as pd
ds_a = load_dataset("theArijitDas/Fake-Reviews-Dataset")
df_a = ds_a["train"].to_pandas()

# Rename columns to standard
df_a = df_a.rename(columns={
    "review": "text",
    "label": "label"
})

# label: 0 = fake, 1 = genuine (already correct)

ds_b = load_dataset("debojit01/fake-review-dataset")
df_b = ds_b["train"].to_pandas()

# Dataset B labels:
# "CG" = Fake, "OR" = Genuine
df_b["label"] = df_b["label"].map({
    "CG": 0,
    "OR": 1
})

df_b = df_b.rename(columns={"review": "text"})

df_a = df_a[["text", "label"]]
df_b = df_b[["text", "label"]]

combined_df = pd.concat([df_a, df_b], ignore_index=True)

from sklearn.utils import shuffle

combined_df = shuffle(combined_df, random_state=42)

print(combined_df["label"].value_counts())
combined_df.to_csv("data/final_fake_review_dataset.csv", index=False)
print("✅ Final dataset saved")
