import streamlit as st
import pandas as pd
import os
import json
from langchain.llms import Ollama
from langchain_experimental.agents import create_csv_agent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---
HISTORY_FILE = "chat_history.json"

# --- Initialize Ollama LLM ---
llm = Ollama(model="qwen2.5-coder:7b")

# --- Functions ---
def load_data():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        return pd.read_csv(uploaded_file, parse_dates=True), uploaded_file
    return None, None


def preprocess_data(df):
    df.fillna("Unknown", inplace=True)
    
    for col in df.select_dtypes(include=["object"]):
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f"{col}_year"] = df[col].dt.year.fillna(0).astype(int)
            df[f"{col}_month"] = df[col].dt.month.fillna(0).astype(int)
        except:
            continue
    
    return df


def basic_analysis(df):
    st.write("### Basic Data Insights")
    st.write("#### Summary Statistics")
    st.write(df.describe(include='all'))
    st.write("#### Missing Values")
    st.write(df.isnull().sum())
    st.write("#### Data Types")
    st.write(df.dtypes)
    
    st.write("#### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def generate_query_and_execute(prompt, agent):
    try:
        response = agent.run(prompt)
        return response
    except Exception as e:
        return f"Error executing query: {e}"


def main():
    st.title("GenAI-Powered CSV Analysis with LangChain")
    data, uploaded_file = load_data()
    history = load_history()
    
    if data is not None:
        data = preprocess_data(data)
        st.write("### Preview of Processed Data", data.head())
        basic_analysis(data)
        
        agent = create_csv_agent(llm, uploaded_file, verbose=True)
        
        prompt = st.text_area("Enter your query:")
        if st.button("Execute Query"):
            result = generate_query_and_execute(prompt, agent)
            st.write("### Query Result")
            st.write(result)
            history.append({"prompt": prompt, "result": result})
            save_history(history)

    if history:
        st.write("### Chat History")
        for item in history:
            st.write(f"**Prompt:** {item['prompt']}")
            st.write(f"**Result:** {item['result']}")
    
    if st.button("Clear History"):
        save_history([])
        st.experimental_rerun()

if __name__ == "__main__":
    main()