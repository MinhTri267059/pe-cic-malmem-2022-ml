"""Chia 200 mẫu x64 (build 19041) thành 100 train / 100 test, phân tầng theo Category."""
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("features_all_x64_19041.csv")
train, test = train_test_split(df, test_size=0.5, random_state=42, stratify=df["Category"])
train.to_csv("x64_train_100.csv", index=False)
test.to_csv("x64_test_100.csv", index=False)

print(f"Train: {len(train)} | Test: {len(test)}")
print(pd.concat([train["Category"].value_counts().rename("train"),
                 test["Category"].value_counts().rename("test")], axis=1))

