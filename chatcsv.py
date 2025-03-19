import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import re
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="Chatbot - Customer Transactions",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables for current query
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to load data
@st.cache_data
def load_data(file_path=None):
    try:
        if file_path is None:
            # In production, you would replace this with your actual file path
            df = pd.read_csv('./data/partial_data_2500.csv')
        else:
            df = pd.read_csv(file_path)
        
        # Convert date columns to datetime format
        date_columns = ['account_open_date', 'last_debit_date', 'last_credit_date', 'date_of_birth']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce', format='%d-%b-%y')
                except:
                    try:
                        # Try another format if the first one fails
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
        
        # Convert boolean columns
        bool_columns = ['mobile_banking', 'internet_banking', 'account_inactive']
        for col in bool_columns:
            if col in df.columns:
                # Convert Yes/No to boolean if needed
                if df[col].dtype == 'object':
                    df[col] = df[col].map({'Yes': True, 'No': False})
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty DataFrame as fallback
        return pd.DataFrame()

# Function to get column descriptions from LLM
def get_column_descriptions(df):
    # Define the URL for the LLM API
    url = "http://localhost:11434/api/generate"
    
    # Get basic column info
    column_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_values = df[col].dropna().sample(min(3, df[col].count())).tolist()
        sample_str = str(sample_values)[:100] + "..." if len(str(sample_values)) > 100 else str(sample_values)
        column_info.append(f"- {col}: {dtype}, example values: {sample_str}")
    
    column_info_str = "\n".join(column_info)
    
    # Create the system prompt for column descriptions
    system_prompt = f"""
    You are a data analyst assistant that helps describe dataset columns.
    Given the following columns with their data types and sample values, provide a brief description for each column.
    Format each description as:
    - column_name: Brief description of what this column represents
    
    For example:
    - account_balance: The current balance in the account
    - internet_banking: Indicates if the customer uses internet banking
    
    Columns information:
    {column_info_str}
    
    Provide ONLY the descriptions, one per line, starting with the column name followed by colon.
    """
    
    # Create the request payload
    payload = {
        "model": "qwen2.5-coder:7b",  # or any other model you have
        "prompt": "Generate detailed descriptions for these database columns",
        "system": system_prompt,
        "stream": False
    }
    
    try:
        # Send the request to LLM
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            # Extract the generated descriptions
            response_data = response.json()
            column_descriptions = response_data.get('response', '')
            
            # Clean up the generated descriptions
            column_descriptions = column_descriptions.strip()
            
            return column_descriptions
        else:
            return "Error getting column descriptions. Using default descriptions."
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"

# Function to get pandas query from LLM
def get_pandas_query(prompt, df_info, column_descriptions):

    # print(f"{column_descriptions}")
    # Define the URL for the LLM API
    url = "http://localhost:11434/api/generate"
    
    # Get dataframe schema
    columns_info = {col: str(df_info[col].dtype) for col in df_info.columns}
    
    # Create the system prompt
    system_prompt = f"""
    You are a data analyst assistant that converts natural language queries to pandas Python code.
    - Only respond with valid Python code for pandas.
    - Do not include any explanation or markdown formatting.
    - The code should start with 'result = ' and return a pandas DataFrame or a calculated value.
    - The dataframe is already loaded as 'df'.
    - Here are the columns and their data types:
    {json.dumps(columns_info, indent=2)}
    
    - Here are descriptions of the columns:
    {column_descriptions}
    
    - Here is the sample data:
    {df_info.head(5).to_markdown()}

    - Use proper pandas syntax and functions.
    - If the query asks for a chart or visualization, create it using plotly and assign to 'result'.
    - If the query asks for statistics or aggregations, calculate them and assign to 'result'.
    - If you're unsure about how to translate the query, create a simple filter that might be helpful.
    - Include adequate related columns in the result.
    - Assume case sensitivity during query formation.
    """
    
    # Create the request payload
    payload = {
        "model": "qwen2.5-coder:7b",  # or any other model you have
        "prompt": f"{prompt}",
        "system": system_prompt,
        "stream": False
    }
    
    try:
        # Send the request to LLM
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            # Extract the generated code from the response
            response_data = response.json()
            generated_code = response_data.get('response', '')
            
            # Clean up the generated code
            generated_code = generated_code.strip()
            # Remove any markdown code blocks if present
            generated_code = re.sub(r'```python\s*', '', generated_code)
            generated_code = re.sub(r'```\s*', '', generated_code)
            
            return generated_code
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"

# Function to execute the pandas query
def execute_pandas_query(query_code, df):
    try:
        # Create a local scope with the dataframe
        local_vars = {'df': df}
        
        # Execute the code in the local scope
        exec(query_code, {}, local_vars)
        
        # Get the result
        result = local_vars.get('result', None)
        
        # Check if result exists or is NaN/None
        if result is None:
            return None, "Some issue has occurred, please rewrite your prompt.", None
        
        # For numeric results, check if NaN
        if isinstance(result, (float, int)) and (pd.isna(result) or result is None):
            return None, "Some issue has occurred, please rewrite your prompt.", None
            
        # For dataframes, check if empty
        if isinstance(result, pd.DataFrame) and result.empty:
            return None, "No matching records found. Please try a different query.", None
        
        # Determine the type of result for appropriate display
        if isinstance(result, pd.DataFrame):
            result_type = "dataframe"
        # Check if result is a plotly figure by looking for common attributes
        elif hasattr(result, 'update_layout') and hasattr(result, 'data'):
            result_type = "plotly_figure"
        elif isinstance(result, (float, int)):
            result_type = "numeric"
        else:
            result_type = "other"

        return result, None, result_type
    except Exception as e:
        return None, "Some issue has occurred, please rewrite your prompt.", None

# Function to display basic info
def show_basic_info(df):

    # Display sample data in the main area
    with st.expander("View Sample Data", expanded=True):
            # Show column selector dropdown
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect(
                "Select columns to display:",
                options=all_columns,
                default=all_columns[:5]  # Default to first 5 columns
    )
            
    # Show sample data with selected columns
    sample_size = st.slider("Number of sample rows:", 3, 20, 5)
    st.dataframe(df[selected_columns].head(sample_size), use_container_width=True)
            
    # Display column data types
    with st.expander("Column Data Types"):
        col_types = pd.DataFrame({
                'Column': df.columns,
                'Data Type': [str(df[col].dtype) for col in df.columns],
                'Sample Value': [str(df[col].iloc[0]) if len(df) > 0 else "" for col in df.columns]
        })
        st.dataframe(col_types, use_container_width=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Total Balance (NPR)", f"{df['account_balance'].sum():,.2f}")
        
    with col2:
        active_accounts = len(df[df['account_inactive'] == False]) if 'account_inactive' in df.columns else "N/A"
        inactive_accounts = len(df[df['account_inactive'] == True]) if 'account_inactive' in df.columns else "N/A"
        
        st.metric("Active Accounts", active_accounts)
        st.metric("Inactive Accounts", inactive_accounts)
    
    # Show distribution of account types if column exists
    if 'account_type' in df.columns:
        st.subheader("Account Types")
        account_types = df['account_type'].value_counts().reset_index()
        account_types.columns = ['Account Type', 'Count']
        fig = px.bar(account_types, x='Account Type', y='Count', color='Count')
        st.plotly_chart(fig)

# Function to check if LLM is running
def llm_connection_status():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            st.success("✅ Connected to LocalLLM")
            return True
        else:
            st.error("❌ LocalLLM is running but returned an error")
            return False
    except:
        st.error("❌ Cannot connect to LocalLLM")
        st.info("Start LocalLLM service to enable query functionality")
        return False

def format_indian_currency(amount):
    """Formats a float value as Indian currency (e.g., 1,65,514.68)."""
    try:
        amount_str = "{:.2f}".format(amount)  # Format to 2 decimal places
        integer_part, decimal_part = amount_str.split('.')

        # Process for Indian numbering system (lakhs, crores)
        s = integer_part[::-1]  # Reverse the string
        groups = [s[0:3]]  # First group of 3 digits
        
        i = 3
        while i < len(s):
            groups.append(s[i:i+2] if i+2 <= len(s) else s[i:])
            i += 2
            
        formatted_integer = ','.join(groups)[::-1]  # Reverse back and join with commas
        
        return f"{formatted_integer}.{decimal_part}"
    except:
        return str(amount)  # Return original amount if formatting fails

        
# Main app
def main():
    st.title("📊Chat FinanceLLM - Transactions")
    
    # Sidebar for options
    st.sidebar.title("Chat Options")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    # In production, you would replace this with your actual file path
    if uploaded_file:
        # Check file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
            
        if file_extension == 'csv':
            df = load_data(uploaded_file)
            if df.empty:
                st.error("Failed to load data. Please check the file path and format.")
                return
            
            # Get column descriptions
            if "column_descriptions" not in st.session_state:
                st.session_state.column_descriptions = get_column_descriptions(df)
            
    # Dashboard and Query buttons
    option_dashboard = st.sidebar.button("Dashboard Overview", use_container_width=True, help="View the dashboard overview")
    option_query = st.sidebar.button("Query Data", use_container_width=True,help="Query the bank data")
    
    # Add LLM connection status indicator
    with st.sidebar:
        st.subheader("", divider=True)
        llm_connected = llm_connection_status()
        
        # Show column descriptions in sidebar
        with st.expander("Column Descriptions"):
            st.markdown(st.session_state.column_descriptions)

    # Main content based on selected option
    if option_dashboard:
        st.header("Bank Data Dashboard")
        st.caption("Overview of the bank data for customer transactions")
        show_basic_info(df)
        
        # Show key metrics
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'mobile_banking' in df.columns:
                mobile_users = len(df[df['mobile_banking'] == True])
                st.metric("Mobile Banking Users", mobile_users, f"{mobile_users/len(df)*100:.1f}%")
            
        with col2:
            if 'internet_banking' in df.columns:
                internet_users = len(df[df['internet_banking'] == True])
                st.metric("Internet Banking Users", internet_users, f"{internet_users/len(df)*100:.1f}%")
            
        with col3:
            if 'account_balance' in df.columns:
                avg_balance = df['account_balance'].mean()
                high_balance = df['account_balance'].max()
                min_balance = df['account_balance'].min()
                st.write("Account Balance")
                st.metric("Avg: ", f"{avg_balance:,.2f}")
                st.metric("Max: ", f"{high_balance:,.2f}")
                st.metric("Min: ", f"{min_balance:,.2f}")
        
        # Show distribution of account categories if column exists
        if 'account_category' in df.columns:
            st.subheader("Account Categories")
            category_amount = df.groupby('account_category')['account_balance'].sum().reset_index()
            category_amount.columns = ['Category', 'Total Balance']
            fig = px.bar(category_amount, x='Category', y='Total Balance', color='Category', orientation='v')
            st.plotly_chart(fig)
        
    # Query mode is the default view
    if option_query or not option_dashboard:
        st.header("Query Your Banking Data")
        
        # Show data sample in sidebar
        with st.expander("View Sample Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.caption("""
        Enter your questions in natural language to analyze the banking data.
        
        Example queries:
        - Show top 10 customers with highest balance
        - Show inactive accounts with a balance greater than 100000
        - Show customers from branch Damauli
        - Plot active vs inactive accounts
        - What is the average balance of the accounts?
        - What is the average balance for branch Damauli?
        - Show average amount for sector 'LOCAL - PERSONS' in Head Office
        - list 5 account holder from damauli with amount > than 500000
        - What is the ratio of active to inactive account
        """)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display data frames, figures, etc.
                if "data" in message:
                    if message["type"] == "dataframe":
                        st.dataframe(message["data"])
                    elif message["type"] == "plotly_figure":
                        st.plotly_chart(message["data"])
                    elif message["type"] == "numeric":
                        st.info(f"Result: {message['data']}")

        # Input for the query
        if query := st.chat_input("Ask something about the data..."):
            # Check if LLM is connected
            if not llm_connected:
                st.error("Cannot process query because LocalLLM is not connected.")
                return
                
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Processing your query..."):
                    # Get the pandas query from LLM
                    pandas_query = get_pandas_query(query, df, st.session_state.column_descriptions)
                    
                    # Display the generated code
                    with st.expander("View Generated Code"):
                        st.code(pandas_query, language="python")
                    
                    # Execute the query
                    result, error, result_type = execute_pandas_query(pandas_query, df)
                    
                    if error:
                        st.error(error)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "type": "error", 
                            "content": f"Error: {error}"
                        })
                    else:
                        if result is None:
                            st.info("No matching records found for your query.")
                            response_content = "No matching records found for your query."
                        elif result_type == "dataframe":
                            st.success(f"Found {len(result)} matching records")
                            st.dataframe(result, use_container_width=True)
                            csv = result.to_csv(index=False)
                            st.download_button(
                                label="Download results as CSV",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv"
                            )
                            response_content = f"Found {len(result)} matching records"
                            # Store the result for message history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "type": "dataframe", 
                                "content": response_content,
                                "data": result
                            })
                        elif result_type == "plotly_figure":
                            st.plotly_chart(result)
                            response_content = "Here's the visualization you requested"
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "type": "plotly_figure", 
                                "content": response_content,
                                "data": result
                            })
                        elif result_type == "numeric":
                            if isinstance(result, float):
                                formatted_result = format_indian_currency(result)
                                st.info(f"Result: {formatted_result}")
                                response_content = f"The result is: {formatted_result}"
                            else:
                                st.info(f"Result: {result}")
                                response_content = f"The result is: {result}"
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "type": "numeric", 
                                "content": response_content,
                                "data": result
                            })
                        else:
                            st.write(result)
                            response_content = str(result)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "type": "text", 
                                "content": response_content
                            })

if __name__ == "__main__":
    main()