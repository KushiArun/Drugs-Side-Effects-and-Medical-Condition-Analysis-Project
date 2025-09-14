# main.py
# Drugs, Side Effects and Medical Condition Analysis Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------
# Step 1: Load Dataset
# ----------------------------
fpath = "drugs_side_effects_drugs_com.csv"   # put your dataset path here
df = pd.read_csv(fpath)

print("Initial Data Shape:", df.shape)
print(df.head())

# ----------------------------
# Step 2: Data Cleaning
# ----------------------------
# Fill missing values safely (NumPy 2.0 fix uses np.nan)
df["side_effects"] = df["side_effects"].fillna("Unknown")
df["related_drugs"] = df["related_drugs"].fillna("Unknown")
df["generic_name"] = df["generic_name"].fillna("Unknown")
df["drug_classes"] = df["drug_classes"].fillna("Unknown")
df["rx_otc"] = df["rx_otc"].fillna("Unknown")
df["pregnancy_category"] = df["pregnancy_category"].fillna("Unknown")
df["rating"] = df["rating"].fillna(0)
df["no_of_reviews"] = df["no_of_reviews"].fillna(0)

# Convert activity column to numeric
df["activity"] = (
    df["activity"].astype(str).str.replace(r"\s+", "", regex=True).str.rstrip("%")
)
df["activity"] = pd.to_numeric(df["activity"], errors="coerce") / 100

# Handle alcohol column
df["alcohol"] = df["alcohol"].replace(np.nan, 0)
df["alcohol"] = df["alcohol"].replace({"X": 1, "0": 0})

# Convert rating and reviews to numeric
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["no_of_reviews"] = pd.to_numeric(df["no_of_reviews"], errors="coerce")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ----------------------------
# Step 3: Exploratory Data Analysis
# ----------------------------

# Distribution of Ratings
plt.figure(figsize=(8, 5))
sns.histplot(df["rating"], bins=10, kde=True)
plt.title("Distribution of Drug Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.savefig("ratings_distribution.png")
plt.close()

# Top 10 Medical Conditions
plt.figure(figsize=(10, 6))
df["medical_condition"].value_counts().head(10).plot(kind="bar", color="skyblue")
plt.title("Top 10 Medical Conditions by Number of Drugs")
plt.xlabel("Medical Condition")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("top_conditions.png")
plt.close()

# Most Common Side Effects
plt.figure(figsize=(10, 6))
df["side_effects"].value_counts().head(10).plot(kind="bar", color="salmon")
plt.title("Top 10 Most Common Side Effects")
plt.xlabel("Side Effect")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("top_side_effects.png")
plt.close()

# Boxplot of Ratings by Drug Class
plt.figure(figsize=(12, 6))
sns.boxplot(x="drug_classes", y="rating", data=df)
plt.xticks(rotation=90)
plt.title("Drug Ratings by Class")
plt.savefig("ratings_by_class.png")
plt.close()

# Correlation Heatmap (numeric columns only)
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# ----------------------------
# Step 4: Feature Encoding + Scaling
# ----------------------------
df_encoded = df.copy()
label_encoder = LabelEncoder()

# Encode categorical features
for col in ["csa", "rx_otc", "generic_name", "medical_condition",
            "pregnancy_category", "side_effects"]:
    df_encoded[col] = label_encoder.fit_transform(df_encoded[col].astype(str))

# Select useful features
features = ["generic_name", "medical_condition", "no_of_reviews", "side_effects",
            "rating", "csa", "pregnancy_category", "rx_otc", "alcohol"]

df_model = df_encoded[features]

# Standardize
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_model), columns=df_model.columns)

# ----------------------------
# Step 5: Save Outputs
# ----------------------------
df.to_csv("cleaned_drugs_dataset.csv", index=False)
df["medical_condition"].value_counts().to_csv("medical_condition_counts.csv")
df["side_effects"].value_counts().to_csv("side_effect_counts.csv")
df["drug_classes"].value_counts().to_csv("drug_classes_counts.csv")

print("\n✅ Analysis Complete!")
print("Outputs saved as:")
print("- cleaned_drugs_dataset.csv")
print("- medical_condition_counts.csv")
print("- side_effect_counts.csv")
print("- drug_classes_counts.csv")
print("- Visualizations: ratings_distribution.png, top_conditions.png, top_side_effects.png, ratings_by_class.png, correlation_heatmap.png")
