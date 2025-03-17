import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M_%S")

# Function to save conversation history
def save_conversation(conversation_id, messages):
    os.makedirs("chatfinance", exist_ok=True)
    with open(f"chatfinance/{conversation_id}.json", "w") as f:
        json.dump(messages, f)

# Function to load conversation history
def load_conversation(conversation_id):
    try:
        with open(f"chatfinance/{conversation_id}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Function to load data
@st.cache_data
def load_data():
    # In production, you would replace this with your actual file path
    df = pd.read_csv('./data/partial_data_2500.csv')
    
    # Convert date columns to datetime format
    date_columns = ['account_open_date', 'last_debit_date', 'last_credit_date', 'date_of_birth']
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce', format='%d-%b-%y')
        except:
            try:
                # Try another format if the first one fails
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    return df

# Function to get pandas query from FinanceLLM
def get_pandas_query(prompt, df_info):
    # Define the URL for the FinanceLLM API
    url = "http://localhost:11434/api/generate"
    
    # Get dataframe schema
    columns_info = {col: str(df_info[col].dtype) for col in df_info.columns}
    
    # Create the system prompt
    system_prompt = f"""
    You are a data analyst assistant that converts natural language queries to pandas Python code.
    - Only respond with valid Python code for pandas.
    - Do not include any explanation or markdown formatting.
    - The code should start with 'result = ' and return a pandas DataFrame.
    - The dataframe is already loaded as 'df'.
    - Here are the columns and their data types:
    {json.dumps(columns_info, indent=2)}
    - Use proper pandas syntax and functions.
    - If the query asks for a chart or visualization, include code to create it using plotly.
    - If you're unsure about how to translate the query, create a simple filter that might be helpful.
    """
    
    # Create the request payload
    payload = {
        "model": "qwen2.5-coder:7b",  # or any other model you have in FinanceLLM
        "prompt": f"{prompt}",
        "system": system_prompt,
        "stream": False
    }
    
    try:
        # Send the request to FinanceLLM
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
        return f"Error connecting to FinanceLLM: {str(e)}"

# Function to execute the pandas query
def execute_pandas_query(query_code, df):
    try:
        # Create a local scope with the dataframe
        local_vars = {'df': df}
        
        # Execute the code in the local scope
        exec(query_code, {}, local_vars)
        
        # Get the result
        result = local_vars.get('result', None)
        
        # If the result is a DataFrame, return it
        if isinstance(result, pd.DataFrame):
            return result, None
        else:
            return None, "The query did not return a DataFrame."
    except Exception as e:
        return None, f"Error executing query: {str(e)}"

# Function to display basic info
def show_basic_info(df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Total Balance (NPR)", f"{df['account_balance'].sum():,.2f}")
        
    with col2:
        st.metric("Active Accounts", len(df[df['account_inactive'] == False]))
        st.metric("Inactive Accounts", len(df[df['account_inactive'] == True]))
    
    st.subheader("Account Types")
    account_types = df['account_type'].value_counts().reset_index()
    account_types.columns = ['Account Type', 'Count']
    fig = px.bar(account_types, x='Account Type', y='Count', color='Count')
    st.plotly_chart(fig)

# Function to check if FinanceLLM is running
def llm_connection_status():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            st.success("✅ Connected to LocalLLM")
                
                # List available models
                # models = response.json().get("models", [])
                # if models:
                #     st.write("Available models:")
                #     for model in models:
                #         st.write(f"- {model['name']}")
                # else:
                #     st.info("No models found.")
                
        else:
            st.error("❌ LocalLLM is running but returned an error")
        return response.status_code == 200
    except:
        st.error("❌ Cannot connect to LocalLLM")
        st.info("Start LocalLLM")
        return False

# Main app
def main():
    st.title("📊Chat FinanceLLM - Transactions")
    
    # Load the data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # # Check FinanceLLM connection
    # llm_connected = llm_connection_status()
    # if not llm_connected:
    #     st.error("⚠️ Cannot connect to FinanceLLM. Please make sure Data is provided to FinanceLLM.")
    #     return
    
    # Sidebar for options
    st.sidebar.title("Options")
    option = st.sidebar.radio(
        "Choose an option:",
        ["Dashboard Overview", "Query Data", "Visualize Data"]
    )
    
    # Show data sample in sidebar
    with st.sidebar.expander("View Sample Data"):
        st.dataframe(df.head(5))
    
    with st.sidebar:
        if st.button("New Chat", use_container_width=True, help="Start a new chat session", icon=":material/chat:"):
            st.session_state.messages = []
            st.session_state.current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M_%S")
            st.rerun()
    
        # Display saved conversations
        st.subheader("Saved Conversations", divider=True)
        if not os.path.exists("chatfinance"):
            os.makedirs("chatfinance")
        saved_conversations = [f.replace(".json", "") for f in os.listdir("chatfinance") if f.endswith(".json")]
        
        if saved_conversations:
            selected_conversation = st.selectbox("Conversation History", saved_conversations)
            if st.button("Load Conversation", type='secondary', use_container_width=True, help="Load a saved conversation", icon=":material/folder_open:"):
                st.session_state.messages = load_conversation(selected_conversation)
                st.session_state.current_conversation_id = selected_conversation
                st.rerun()
    
        # Add LocalLLM connection status indicator
        st.subheader("", divider=True)
        llm_connection_status()

    # Main content based on selected option
    if option == "Dashboard Overview":
        st.header("Bank Data Dashboard")
        st.caption("Overview of the bank data for some transactions. To be adjusted.")
        show_basic_info(df)
        
        # Show some key metrics
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mobile_users = len(df[df['mobile_banking'] == True])
            st.metric("Mobile Banking Users", mobile_users, f"{mobile_users/len(df)*100:.1f}%")
            internet_users = len(df[df['internet_banking'] == True])
            st.metric("Internet Banking Users", internet_users, f"{internet_users/len(df)*100:.1f}%")
            
        with col2:
            internet_users = len(df[df['internet_banking'] == True])
            st.metric("Internet Banking Users", internet_users, f"{internet_users/len(df)*100:.1f}%")
            
        with col3:
            avg_balance = df['account_balance'].mean()
            high_balance = df['account_balance'].max()
            min_balance = df['account_balance'].min()
            st.write("Account Balance")
            st.metric("Avg: ",f"{avg_balance:,.2f}")
            st.metric("Max: ",f"{high_balance:,.2f}")
            st.metric("Min: ",f"{min_balance:,.2f}")
        
        # Show distribution of account categories
        st.subheader("Account Categories")
        category_amount = df.groupby('account_category')['account_balance'].sum().reset_index()
        category_amount.columns = ['Category', 'Total Balance']
        fig = px.bar(category_amount, x='Category', y='Total Balance', color='Category', orientation='v')
        st.plotly_chart(fig)

        # st.subheader("Account Types")
        # account_types = df['account_type'].value_counts().reset_index()
        # account_types.columns = ['Account Type', 'Count']
        # fig = px.bar(account_types, x='Account Type', y='Count', color='Count')
        # st.plotly_chart(fig)
        
    elif option == "Query Data":
        st.caption("""
        ---
        ### Query the Bank Data:
        1. Enter your question in natural language, prompt.
            - Ask for code generation, chat completion or necessary details by describing what you need
            - Show top 10 customers with highest balance
            - Show inactive accounts with a balance greater than 1000000
            - Show customers from branch Damauli
        """)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input for the query
        # query = st.chat_input("Your Query", placeholder="e.g., Show inactive accounts with balance greater than 1000000")
        if query := st.chat_input("Show inactive accounts with balance greater than 1000000"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})  #Prompt is the user input
    
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Processing your query with FinanceLLM..."):
                    # Get the pandas query from FinanceLLM
                    pandas_query = get_pandas_query(query, df)
                    
                    # Display the generated code
                    with st.expander("View Generated Code"):
                        st.code(pandas_query, language="python")
                    
                    # Execute the query
                    result, error = execute_pandas_query(pandas_query, df)
                    
                    if error:
                        st.error(error)
                    elif result is not None and not result.empty:
                        st.success(f"Found {len(result)} matching records")
                        st.dataframe(result, use_container_width=True)
                        
                        # # Highlight key columns
                        # if 'account_balance' in result.columns:
                        #     st.subheader("Account Balance Distribution")
                        #     fig = px.histogram(result, x='account_balance', nbins=20)
                        #     st.plotly_chart(fig)
                        
                        # Download option
                        csv = result.to_csv(index=False)
                        st.download_button(
                            label="Download results as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv"
                        )
                    elif result is not None:
                        st.info("No matching records found for your query.")
                        st.warning("FinanceLLM couldn't understand that query. Here are some examples you can try:\n" +
                    "- Show inactive accounts\n" +
                    "- Show customers with mobile banking\n" +
                    "- Show accounts with balance greater than 100000\n" +
                    "- Show top 10 customers with highest balance\n" +
                    "- Show customers from Damauli\n" +
                    "- Show staff accounts")
                    
                    save_conversation(st.session_state.current_conversation_id, st.session_state.messages)

    elif option == "Visualize Data":
        st.header("Visualize Bank Data")
        st.write("Enter what you want to visualize and FinanceLLM will generate a plot.")
        
        viz_query = st.text_input("Visualization Request", placeholder="e.g., Create a bar chart of account types by branch")
        
        if viz_query:
            with st.spinner("Creating visualization with FinanceLLM..."):
                # Get the pandas query from FinanceLLM
                viz_code = get_pandas_query(f"Create a visualization that {viz_query}", df)
                
                # Display the generated code
                with st.expander("View Generated Visualization Code"):
                    st.code(viz_code, language="python")
                
                # Execute the visualization code
                try:
                    # Create a local scope with the dataframe and plotly
                    local_vars = {'df': df, 'px': px}
                    
                    # Execute the code in the local scope
                    exec(viz_code, {}, local_vars)
                    
                    # Get the result
                    result = local_vars.get('result', None)
                    
                    if 'fig' in local_vars:
                        st.plotly_chart(local_vars['fig'])
                    elif result is not None:
                        st.dataframe(result)
                    else:
                        st.warning("No visualization was generated.")
                except Exception as e:
                    st.error(f"Error executing visualization code: {str(e)}")
        
        # Always show some default visualizations
        st.subheader("Account Activity")
        active_vs_inactive = df['account_inactive'].value_counts().reset_index()
        active_vs_inactive.columns = ['Status', 'Count']
        active_vs_inactive['Status'] = active_vs_inactive['Status'].map({True: 'Inactive', False: 'Active'})
        fig = px.pie(active_vs_inactive, values='Count', names='Status', title='Active vs Inactive Accounts')
        st.plotly_chart(fig)
        
        # Banking services adoption
        st.subheader("Banking Services Adoption")
        services = pd.DataFrame({
            'Service': ['Mobile Banking', 'Internet Banking', 'Account Service'],
            'Users': [
                df['mobile_banking'].sum(),
                df['internet_banking'].sum(),
                df['account_service'].sum()
            ]
        })
        fig = px.bar(services, x='Service', y='Users', color='Service')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()