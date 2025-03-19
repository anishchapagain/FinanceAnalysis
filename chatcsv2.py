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
    
# Initialize data loading state
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    
if "df" not in st.session_state:
    st.session_state.df = None
    
if "column_descriptions" not in st.session_state:
    st.session_state.column_descriptions = ""

# Function to load data from uploaded file
def load_uploaded_file(uploaded_file):
    try:
        # Check file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        else:
            return None, f"Unsupported file format: {file_extension}. Please upload a CSV or Excel file."
        
        # Basic data cleaning
        # Convert date columns (detect based on column name containing 'date')
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Convert common boolean columns
        bool_columns = [col for col in df.columns if any(term in col.lower() for term in ['is_', 'has_', '_flag', 'active', 'enabled', 'status'])]
        for col in bool_columns:
            if df[col].dtype == 'object':
                # Try to convert common boolean representations
                try:
                    df[col] = df[col].map({'Yes': True, 'No': False, 'yes': True, 'no': False, 
                                           'Y': True, 'N': False, 'TRUE': True, 'FALSE': False,
                                           'true': True, 'false': False, '1': True, '0': False})
                except:
                    pass
        
        # Success
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

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
    - Assume case sensitivity during query formation.
    - Handle boolean columns correctly (True/False, not 'Yes'/'No').
    - If the query requires complex operations, break it down into steps.
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
    col1, col2 = st.columns(2)
    
    # Try to identify key columns based on column names
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    with col1:
        st.metric("Total Records", len(df))
        if numeric_cols:
            # Find first numeric column that might represent money
            money_cols = [col for col in numeric_cols 
                         if any(term in col.lower() for term in ['amount', 'balance', 'price', 'cost', 'value', 'total'])]
            if money_cols:
                st.metric(f"Total {money_cols[0]}", f"{df[money_cols[0]].sum():,.2f}")
            else:
                st.metric(f"Total {numeric_cols[0]}", f"{df[numeric_cols[0]].sum():,.2f}")
        
    with col2:
        # Look for status columns
        status_cols = [col for col in df.columns 
                      if any(term in col.lower() for term in ['status', 'active', 'inactive', 'enabled'])]
        
        if status_cols:
            if df[status_cols[0]].dtype == 'bool':
                active_count = df[df[status_cols[0]] == True].shape[0]
                inactive_count = df[df[status_cols[0]] == False].shape[0]
                
                st.metric(f"Active ({status_cols[0]})", active_count)
                st.metric(f"Inactive ({status_cols[0]})", inactive_count)
        else:
            # If no status column found, show other metrics
            if len(numeric_cols) >= 2:
                st.metric(f"Average {numeric_cols[0]}", f"{df[numeric_cols[0]].mean():,.2f}")
                st.metric(f"Max {numeric_cols[0]}", f"{df[numeric_cols[0]].max():,.2f}")
    
    # Show distribution of a categorical column if one exists
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        selected_cat = categorical_cols[0]  # Use first categorical column
        if df[selected_cat].nunique() < 15:  # Only if reasonable number of categories
            st.subheader(f"Distribution of {selected_cat}")
            counts = df[selected_cat].value_counts().reset_index()
            counts.columns = [selected_cat, 'Count']
            fig = px.bar(counts, x=selected_cat, y='Count', color=selected_cat)
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
    st.sidebar.title("Data Loading")
    
    # File uploader in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    # Load demo data button
    use_demo_data = st.sidebar.checkbox("Use demo data instead")
    
    if use_demo_data:
        try:
            # Try to load demo data
            st.sidebar.text("Loading demo data...")
            try:
                demo_df = pd.read_csv('./data/partial_data_2500.csv')
                st.sidebar.success(f"Demo data loaded: {len(demo_df)} records")
                st.session_state.df = demo_df
                st.session_state.data_loaded = True
                
                # Reset column descriptions to force regeneration
                st.session_state.column_descriptions = ""
            except Exception as e:
                st.sidebar.error(f"Could not load demo data: {str(e)}")
        except:
            st.sidebar.error("Demo data file not found")
    
    elif uploaded_file is not None:
        # Process the uploaded file
        st.sidebar.text("Loading data...")
        df, error = load_uploaded_file(uploaded_file)
        
        if error:
            st.sidebar.error(error)
        else:
            # Store the dataframe in session state
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            # Reset column descriptions to force regeneration
            st.session_state.column_descriptions = ""
            
            st.sidebar.success(f"File loaded successfully: {len(df)} records")
            st.sidebar.info(f"Columns: {', '.join(df.columns)}")
    
    # Data operations section (only show if data is loaded)
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.sidebar.title("Options")
        
        # Show data summary in sidebar
        with st.sidebar.expander("Data Summary"):
            df = st.session_state.df
            st.write(f"Records: {len(df)}")
            st.write(f"Columns: {len(df.columns)}")
            if len(df) > 0:
                missing_data = df.isnull().sum().sum()
                st.write(f"Missing values: {missing_data}")
        
        # Dashboard and Query buttons
        option_dashboard = st.sidebar.button("Dashboard Overview", use_container_width=True)
        option_query = st.sidebar.button("Query Data", use_container_width=True)
        
        # Add LLM connection status indicator
        with st.sidebar:
            st.subheader("", divider=True)
            llm_connected = llm_connection_status()
            
            # Generate column descriptions if not already done
            if not st.session_state.column_descriptions:
                spinner_placeholder = st.empty()
                spinner_placeholder.text("Analyzing dataset structure...")
                st.session_state.column_descriptions = get_column_descriptions(st.session_state.df)
                spinner_placeholder.empty()
            
            # Show column descriptions in sidebar
            with st.expander("Column Descriptions"):
                st.markdown(st.session_state.column_descriptions)
        
        # Main content based on selected option
        if option_dashboard:
            st.header("Bank Data Dashboard")
            st.caption("Overview of the data")
            show_basic_info(st.session_state.df)
            
            # Show key metrics
            df = st.session_state.df
            st.subheader("Key Metrics")
            
            # Dynamically identify numeric and categorical columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            # Display metrics based on available columns
            if numeric_cols:
                cols = st.columns(min(3, len(numeric_cols)))
                for i, col in enumerate(numeric_cols[:3]):  # Show up to 3 numeric columns
                    with cols[i % 3]:
                        st.metric(f"{col}", f"{df[col].mean():.2f}", 
                                  f"Max: {df[col].max():.2f}")
            
            # Show visualizations for categorical columns
            if categorical_cols:
                for col in categorical_cols[:2]:  # Show up to 2 categorical visualizations
                    st.subheader(f"Distribution of {col}")
                    if df[col].nunique() <= 10:  # Only show if reasonable number of categories
                        counts = df[col].value_counts().reset_index()
                        counts.columns = [col, 'Count']
                        fig = px.bar(counts, x=col, y='Count', color=col)
                        st.plotly_chart(fig)
        
        # Query mode is the default view or when selected
        if option_query or not option_dashboard:
            # Check if LLM is connected before showing query interface
            if not llm_connected:
                st.warning("LocalLLM is not connected. Please start the LocalLLM service to enable querying.")
                st.info("You can still explore the data in the dashboard view.")
            else:
                st.header("Query Your Banking Data")
                
                # Display sample data in the main area
                with st.expander("View Sample Data", expanded=True):
                    df = st.session_state.df
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
                
                # Add a column explorer for easier query formulation
                with st.expander("Column Explorer"):
                    # Allow selecting a specific column to explore
                    selected_column = st.selectbox("Select a column to explore:", df.columns.tolist())
                    
                    if selected_column:
                        # Show basic statistics for the selected column
                        col1, col2 = st.columns(2)
                        with col1:
                            if df[selected_column].dtype in ['int64', 'float64']:
                                st.write("Numeric Statistics:")
                                stats = df[selected_column].describe()
                                st.dataframe(stats)
                            elif df[selected_column].dtype == 'bool':
                                st.write("Boolean Distribution:")
                                bool_counts = df[selected_column].value_counts()
                                st.dataframe(bool_counts)
                            else:
                                st.write("Top Values:")
                                value_counts = df[selected_column].value_counts().head(10)
                                st.dataframe(value_counts)
                        
                        with col2:
                            # Show a simple visualization of the column
                            st.write("Quick Visualization:")
                            try:
                                if df[selected_column].dtype in ['int64', 'float64']:
                                    # Histogram for numeric data
                                    fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
                                    st.plotly_chart(fig, use_container_width=True)
                                elif df[selected_column].dtype == 'bool':
                                    # Pie chart for boolean data
                                    fig = px.pie(df, names=selected_column, title=f"Distribution of {selected_column}")
                                    st.plotly_chart(fig, use_container_width=True)
                                elif df[selected_column].nunique() < 15:
                                    # Bar chart for categorical data with few unique values
                                    fig = px.bar(df[selected_column].value_counts().reset_index(), 
                                                x='index', y=selected_column, 
                                                title=f"Counts of {selected_column}")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.write("Too many unique values for visualization")
                            except Exception as e:
                                st.error(f"Could not visualize: {str(e)}")

                # Add query suggestions based on dataset
                st.markdown("### Query Suggestions")
                
                # Generate dynamic suggestions based on available columns
                suggestions = []
                df = st.session_state.df
                
                # Add general suggestions
                suggestions.append("Show the first 10 rows of data")
                
                # Add suggestions based on numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    suggestions.append(f"What is the average {numeric_cols[0]}?")
                    suggestions.append(f"Show top 10 records with highest {numeric_cols[0]}")
                    
                # Add suggestions based on categorical columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_cols:
                    suggestions.append(f"Show distribution of {categorical_cols[0]}")
                    if len(categorical_cols) > 1 and len(numeric_cols) > 0:
                        suggestions.append(f"Show average {numeric_cols[0]} by {categorical_cols[0]}")
                
                # Add boolean column suggestions
                bool_cols = [col for col in df.columns if df[col].dtype == 'bool']
                if bool_cols:
                    suggestions.append(f"Show records where {bool_cols[0]} is True")
                
                # Create clickable suggestion buttons
                col1, col2 = st.columns(2)
                with col1:
                    for i in range(0, len(suggestions), 2):
                        if st.button(suggestions[i], key=f"sugg_{i}", use_container_width=True):
                            query = suggestions[i]
                with col2:
                    for i in range(1, len(suggestions), 2):
                        if i < len(suggestions):
                            if st.button(suggestions[i], key=f"sugg_{i}", use_container_width=True):
                                query = suggestions[i]
                
                st.caption("""
                Enter your questions in natural language to analyze the data.
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
                query = st.chat_input("Ask something about the data...")
                if query:
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
                            pandas_query = get_pandas_query(query, st.session_state.df, st.session_state.column_descriptions)
                            
                            # Display the generated code
                            with st.expander("View Generated Code"):
                                st.code(pandas_query, language="python")
                            
                            # Execute the query
                            result, error, result_type = execute_pandas_query(pandas_query, st.session_state.df)
                            
                            if error:
                                st.error(error)
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "type": "error", 
                                    "content": "Some issue has occurred, please rewrite your prompt."
                                })
                            else:
                                if result is None:
                                    st.info("Some issue has occurred, please rewrite your prompt.")
                                    response_content = "Some issue has occurred, please rewrite your prompt."
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "type": "error", 
                                        "content": response_content
                                    })
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
    else:
        # Show instructions if no data is loaded
        st.info("👈 Please upload a CSV or Excel file in the sidebar or use the demo data to get started.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("How to use this app")
            st.markdown("""
            1. **Upload Data**: Use the file uploader in the sidebar to upload your CSV or Excel file.
            2. **Explore Data**: Once uploaded, you can view a dashboard of key metrics or query the data.
            3. **Ask Questions**: Type natural language questions about your data to get insights.
            """)
        
        with col2:
            st.subheader("Example Queries")
            st.markdown("""
            Once you upload data, you can ask questions like:
            - "Show me the top 10 records with highest values"
            - "What is the average of [column]?"
            - "Plot the distribution of [column]"
            - "Show records where [column] equals [value]"
            - "Calculate the sum of [column] grouped by [category]"
            """)
            
        # Add some sample images or illustrations
        st.markdown("---")
        st.subheader("Features")
        
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        with feat_col1:
            st.markdown("### 📊 Data Visualization")
            st.markdown("Automatically visualize your data with charts and graphs.")
        with feat_col2:
            st.markdown("### 🤖 Natural Language Queries")
            st.markdown("Ask questions about your data in plain English.")
        with feat_col3:
            st.markdown("### 📈 Data Analysis")
            st.markdown("Get insights and statistics from your data instantly.")

if __name__ == "__main__":
    main()