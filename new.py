import streamlit as st
import pandas as pd
from ollama import Ollama

# Initialize the Ollama model
ollama = Ollama()

# Define the Streamlit app
def main():
    st.title("Pandas CSV Processor")

    # Upload the CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Display the uploaded CSV file
        st.subheader("Uploaded CSV File:")
        st.write(df)

        # Get the user prompt
        user_prompt = st.text_input(label="Enter a prompt (e.g., 'get the first 5 rows')")

        if user_prompt:
            # Convert the user prompt to Pandas code using Ollama
            code = ollama.convert(user_prompt)

            # Display the generated code
            st.subheader("Generated Code:")
            st.code(code)

            # Execute the code
            try:
                result = eval(code, {"df": df, "pd": pd})
                st.subheader("Result:")
                st.write(result)
            except Exception as e:
                # Handle any exceptions
                st.error(f"Error: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()