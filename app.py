
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Data Quality Checker", layout="wide")
st.title("📊 AI-Powered Data Quality Checker")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Uploaded Data")
    st.dataframe(df)

    st.subheader("📌 Data Quality Report")
    report = []

    # Check for missing values
    missing_values = df.isnull().sum()
    for col, count in missing_values.items():
        if count > 0:
            report.append(f"Column '{col}' has {count} missing values.")

    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        report.append(f"There are {duplicate_count} duplicate rows.")

    # Check for numerical outliers using IQR
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        if not outliers.empty:
            report.append(f"Column '{col}' has {len(outliers)} potential outliers.")

    # Display report
    if report:
        for item in report:
            st.warning(item)
    else:
        st.success("No major data quality issues found.")

    # Basic visualizations
    st.subheader("📈 Visualizations")
    selected_col = st.selectbox("Select a column to visualize", df.columns)
    if df[selected_col].dtype == 'object':
        st.bar_chart(df[selected_col].value_counts())
    else:
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

    # Generate AI summary
    st.subheader("🤖 AI Summary")
    if st.button("Generate AI Report"):
        with st.spinner("Generating AI report..."):
            prompt = f"Generate a summary report on the following employee data quality issues and overall data insights:\n\n{df.head(10).to_markdown()}"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            ai_summary = response.choices[0].message.content
            st.info(ai_summary)
