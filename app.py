# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# Load the cleaned dataset
# ----------------------------
df = pd.read_csv("cleaned_drugs_dataset.csv")

st.set_page_config(page_title="Drug Analysis Dashboard", layout="wide")
st.title("💊 Drugs, Side Effects & Medical Conditions Dashboard")

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 Filters")
condition = st.sidebar.selectbox("Select Medical Condition", ["All"] + sorted(df["medical_condition"].unique().tolist()))
drug_class = st.sidebar.selectbox("Select Drug Class", ["All"] + sorted(df["drug_classes"].unique().tolist()))

filtered_df = df.copy()
if condition != "All":
    filtered_df = filtered_df[filtered_df["medical_condition"] == condition]
if drug_class != "All":
    filtered_df = filtered_df[filtered_df["drug_classes"] == drug_class]

st.write(f"### Showing {len(filtered_df)} records")

st.dataframe(filtered_df.head(20))

# ----------------------------
# Charts
# ----------------------------
st.subheader("📊 Data Insights")

# Distribution of Ratings
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(filtered_df["rating"], bins=10, kde=True, ax=ax)
ax.set_title("Distribution of Ratings")
st.pyplot(fig)

# Top Medical Conditions
st.subheader("Top Medical Conditions")
top_conditions = df["medical_condition"].value_counts().head(10)
st.bar_chart(top_conditions)

# Most Common Side Effects
st.subheader("Top Reported Side Effects")
top_side_effects = df["side_effects"].value_counts().head(10)
st.bar_chart(top_side_effects)

# Ratings by Drug Class
st.subheader("Drug Ratings by Class")
fig, ax = plt.subplots(figsize=(10,4))
sns.boxplot(x="drug_classes", y="rating", data=filtered_df, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)
