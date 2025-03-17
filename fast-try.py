from fastapi import FastAPI, UploadFile, File
import pandas as pd
import ollama
import json
from io import StringIO
import streamlit as st
import requests

app = FastAPI()
datasets = {}

@app.post("/upload/{file_name}")
async def upload_csv(file_name: str, file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))
    datasets[file_name] = df
    return {"message": f"{file_name} uploaded successfully", "columns": df.columns.tolist()}

@app.post("/query")
async def query_llm(prompt: str):
    response = ollama.chat(
        model="gemma3:latest", # "qwen2.5-coder:7b",
        messages=[{"role": "user", "content": f"Dataset: {json.dumps({k: v.head(3).to_dict() for k, v in datasets.items()})}\nQuery: {prompt}\nReturn Python Pandas code."}]
    )
    code = response["message"]["content"]
    try:
        exec_locals = {}
        exec(code, {"pd": pd, "datasets": datasets}, exec_locals)
        result = exec_locals.get("result", "No result variable found.")
        return {"query": prompt, "result": result.to_dict() if isinstance(result, pd.DataFrame) else result}
    except Exception as e:
        return {"error": str(e), "code": code}

if __name__ == "__main__":
    import uvicorn
    import threading

    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    thread = threading.Thread(target=run_api)
    thread.start()

    st.title("Banking Chatbot")
    st.sidebar.header("Upload CSV Files")
    uploaded_files = st.sidebar.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            response = requests.post(f"http://localhost:8000/upload/{uploaded_file.name}", files={"file": uploaded_file})
            st.sidebar.write(response.json()["message"])
    
    st.header("Ask a question about your data")
    user_query = st.text_input("Enter your query:")
    if st.button("Submit"):
        response = requests.post("http://localhost:8000/query", json={"prompt": user_query})
        result = response.json()
        print(result)
        if "error" in result:
            st.error(result["error"])
        else:
            st.write(result["result"])
    
    thread.join()
