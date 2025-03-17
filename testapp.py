import streamlit as st
import pandas as pd
import json
import requests
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import plotly.express as px
from typing import Dict, Any, List, Tuple

class BankDataAnalyzer:
    def __init__(self, df):
        """Initialize with a dataframe"""
        self.raw_df = df.copy()
        self.df = None
        self.analysis_results = {}
        self.processed = False
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        df = self.raw_df.copy()
        
        # Convert date columns to datetime
        date_columns = ['account_open_date', 'last_debit_date', 'last_credit_date', 'date_of_birth']
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', format='%d-%b-%y')
        
        # Convert boolean columns
        bool_columns = ['mobile_banking', 'internet_banking', 'account_service', 'kyc_status', 'account_inactive']
        for col in bool_columns:
            df[col] = df[col].map({'True': True, 'False': False}) if df[col].dtype == 'object' else df[col]
        
        # Clean and convert numeric columns
        numeric_columns = ['account_balance', 'local_currency_balance']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.df = df
        return df
    
    def engineer_features(self):
        """Create new features from existing data"""
        if self.df is None:
            self.preprocess_data()
            
        df = self.df.copy()
        
        # Calculate account age in years
        today = pd.Timestamp.now()
        df['account_age_years'] = ((today - df['account_open_date']).dt.days / 365.25).round(2)
        
        # Calculate customer age in years
        df['customer_age'] = ((today - df['date_of_birth']).dt.days / 365.25).round()
        
        # Calculate days since last transaction
        df['days_since_last_debit'] = (today - df['last_debit_date']).dt.days
        df['days_since_last_credit'] = (today - df['last_credit_date']).dt.days
        
        # Calculate transaction recency (min of last debit and credit)
        df['transaction_recency_days'] = df[['days_since_last_debit', 'days_since_last_credit']].min(axis=1)
        
        # Account activity level
        df['activity_level'] = pd.cut(
            df['transaction_recency_days'],
            bins=[0, 30, 90, 180, 365, float('inf')],
            labels=['Very Active', 'Active', 'Moderate', 'Inactive', 'Dormant']
        )
        
        # Customer segments based on balance
        df['balance_segment'] = pd.cut(
            df['account_balance'],
            bins=[0, 1000, 10000, 100000, 1000000, float('inf')],
            labels=['Minimal', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Digital engagement score
        df['digital_engagement'] = df['mobile_banking'].astype(int) + df['internet_banking'].astype(int)
        
        # Account health score (composite)
        df['account_health'] = (
            (df['account_balance'] > 0).astype(int) * 2 +
            (df['transaction_recency_days'] < 90).astype(int) * 2 +
            df['digital_engagement'] +
            (~df['account_inactive']).astype(int) * 3
        )
        
        self.df = df
        self.processed = True
        return df
    
    def perform_analysis(self):
        """Perform comprehensive analysis on the data"""
        if not self.processed:
            self.engineer_features()
            
        df = self.df
        results = {}
        
        # Balance distribution
        results['balance_stats'] = {
            'mean': df['account_balance'].mean(),
            'median': df['account_balance'].median(),
            'min': df['account_balance'].min(),
            'max': df['account_balance'].max(),
            'std': df['account_balance'].std(),
            'distribution': df['balance_segment'].value_counts().to_dict()
        }
        
        # Customer demographics
        results['demographics'] = {
            'avg_age': df['customer_age'].mean(),
            'nationality_distribution': df['nationality'].value_counts().to_dict(),
            'residency_distribution': df['residency_status'].value_counts().to_dict(),
            'industry_distribution': df['customer_industry'].value_counts().head(10).to_dict()
        }
        
        # Account activity
        results['account_activity'] = {
            'avg_account_age': df['account_age_years'].mean(),
            'avg_transaction_recency': df['transaction_recency_days'].mean(),
            'activity_distribution': df['activity_level'].value_counts().to_dict()
        }
        
        # Digital engagement
        results['digital_engagement'] = {
            'mobile_banking_pct': df['mobile_banking'].mean() * 100,
            'internet_banking_pct': df['internet_banking'].mean() * 100,
            'engagement_distribution': df['digital_engagement'].value_counts().to_dict()
        }
        
        # Account health
        results['account_health'] = {
            'avg_health_score': df['account_health'].mean(),
            'inactive_accounts_pct': df['account_inactive'].mean() * 100,
            'health_distribution': df.groupby(pd.cut(df['account_health'], 
                                                    bins=[0, 3, 6, 9, float('inf')],
                                                    labels=['Poor', 'Fair', 'Good', 'Excellent']))['customer_id'].count().to_dict()
        }
        
        # Branch analysis
        results['branch_analysis'] = {
            'accounts_by_branch': df['bank_branch'].value_counts().to_dict(),
            'avg_balance_by_branch': df.groupby('bank_branch')['account_balance'].mean().to_dict()
        }
        
        # Account types
        results['account_types'] = {
            'distribution': df['account_type'].value_counts().to_dict(),
            'avg_balance_by_type': df.groupby('account_type')['account_balance'].mean().to_dict()
        }
        
        self.analysis_results = results
        return results

    def get_insights(self) -> List[str]:
        """Generate insights from the analysis"""
        if not self.analysis_results:
            self.perform_analysis()
            
        insights = []
        results = self.analysis_results
        df = self.df
        
        # Balance insights
        if results['balance_stats']['median'] < results['balance_stats']['mean'] * 0.5:
            insights.append("The account balance distribution is skewed, with a few high-value accounts pulling up the average.")
        
        # Activity insights
        inactive_pct = df['activity_level'].isin(['Inactive', 'Dormant']).mean() * 100
        if inactive_pct > 25:
            insights.append(f"A significant portion ({inactive_pct:.1f}%) of accounts are inactive or dormant.")
        
        # Digital engagement insights
        if results['digital_engagement']['mobile_banking_pct'] > results['digital_engagement']['internet_banking_pct']:
            insights.append("Mobile banking adoption is higher than internet banking, suggesting a preference for mobile services.")
        
        # Account health insights
        poor_health_pct = df['account_health'].lt(3).mean() * 100
        if poor_health_pct > 20:
            insights.append(f"About {poor_health_pct:.1f}% of accounts are in poor health and may need attention.")
        
        # Branch insights
        branches = list(results['branch_analysis']['accounts_by_branch'].keys())
        if len(branches) > 1:
            main_branch = max(results['branch_analysis']['accounts_by_branch'], key=results['branch_analysis']['accounts_by_branch'].get)
            main_branch_pct = results['branch_analysis']['accounts_by_branch'][main_branch] / len(df) * 100
            insights.append(f"The {main_branch} branch holds {main_branch_pct:.1f}% of all accounts.")
        
        # Account type insights
        top_account_type = max(results['account_types']['distribution'], key=results['account_types']['distribution'].get)
        top_type_pct = results['account_types']['distribution'][top_account_type] / len(df) * 100
        insights.append(f"The most common account type is '{top_account_type}' ({top_type_pct:.1f}% of accounts).")
        
        # Balance segment insights
        high_balance_pct = df['balance_segment'].isin(['High', 'Very High']).mean() * 100
        insights.append(f"About {high_balance_pct:.1f}% of accounts have high or very high balances.")
        
        return insights
    
    def generate_visualizations(self):
        """Generate key visualizations for the data"""
        if not self.processed:
            self.engineer_features()
            
        df = self.df
        visualizations = {}
        
        # Balance distribution
        fig1 = px.histogram(df, x='account_balance', nbins=50, title='Account Balance Distribution')
        fig1.update_layout(xaxis_title='Account Balance', yaxis_title='Number of Accounts')
        visualizations['balance_distribution'] = fig1
        
        # Account types
        fig2 = px.bar(df['account_type'].value_counts().reset_index(), 
                     x='index', y='account_type', title='Account Types')
        fig2.update_layout(xaxis_title='Account Type', yaxis_title='Number of Accounts')
        visualizations['account_types'] = fig2
        
        # Activity level
        fig3 = px.pie(df, names='activity_level', title='Account Activity Levels')
        visualizations['activity_levels'] = fig3
        
        # Digital engagement
        digital_data = pd.DataFrame({
            'Engagement Type': ['Mobile Banking', 'Internet Banking'],
            'Percentage': [
                df['mobile_banking'].mean() * 100,
                df['internet_banking'].mean() * 100
            ]
        })
        fig4 = px.bar(digital_data, x='Engagement Type', y='Percentage', 
                     title='Digital Engagement')
        visualizations['digital_engagement'] = fig4
        
        # Balance by branch
        fig5 = px.box(df, x='bank_branch', y='account_balance', 
                     title='Account Balance by Branch')
        visualizations['balance_by_branch'] = fig5
        
        # Account health
        health_bins = pd.cut(df['account_health'], 
                            bins=[0, 3, 6, 9, float('inf')],
                            labels=['Poor', 'Fair', 'Good', 'Excellent'])
        fig6 = px.pie(health_bins.value_counts().reset_index(), 
                     names='index', values='count', 
                     title='Account Health Distribution')
        visualizations['account_health'] = fig6
        
        return visualizations
    
class CSVChatBot:
    def __init__(self, df=None):
        self.df = None
        self.raw_df = df
        self.analyzer = None
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Initialize the analyzer if df is provided
        if df is not None:
            self.setup_analyzer()
        
        
    def load_csv(self, file):
        """Load a CSV file into pandas DataFrame"""
        try:
            self.raw_df = pd.read_csv(file)
            self.setup_analyzer()
            return f"CSV file loaded successfully. Found {len(self.df)} rows and {len(self.df.columns)} columns."
        except Exception as e:
            return f"Error loading CSV: {str(e)}"
    
    def setup_analyzer(self):
        """Set up the analyzer with the dataframe"""
        self.analyzer = BankDataAnalyzer(self.raw_df)
        # Process the data and store the processed dataframe
        self.df = self.analyzer.engineer_features()
        print("Data processed successfully.")
        print(self.df.columns)
        # Perform analysis
        results = self.analyzer.perform_analysis()
        return results

    def get_df_info(self):
        """Get information about the DataFrame"""
        if self.df is None:
            return "No CSV file loaded yet."
        
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in zip(self.df.columns, self.df.dtypes)},
            "sample": self.df.head(3).to_dict(orient='records')
        }
        return info
    
    def prompt_to_query(self, prompt):
        """Convert natural language prompt to pandas query using Ollama"""
        if self.df is None:
            return None, "No CSV file loaded yet."
        
        df_info = self.get_df_info()
        
        # Create a prompt for Ollama
        system_prompt = (
            "You are a data analysis assistant that converts natural language to pandas Python code. "
            "Given the DataFrame information and a user question, generate ONLY the pandas Python code "
            "needed to answer the question. Return ONLY valid Python code without explanation or markdown. "
            "DO NOT use 'print()' functions as they won't work in the execution environment. "
            "Instead, make sure your code ends by returning the result directly, using expressions "
            "that evaluate to the final result. For example, just write 'df.head()' not 'print(df.head())'." \
            "Use currency code 'NPR' unless specified otherwise."
        )
        
        user_prompt = f"""
DataFrame Information:
- Shape: {df_info['shape']}
- Columns: {df_info['columns']}
- Data Types: {df_info['dtypes']}
- Sample data: {json.dumps(df_info['sample'], indent=2)}

User question: {prompt}

Convert this question to pandas code that will answer the question. Return ONLY the Python code.
IMPORTANT: DO NOT use any print() functions as they will cause errors.
Simply return the result directly (e.g., 'df.head()' not 'print(df.head())').
"""
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    # "model": "qwen2.5:latest",
                    "model": "qwen2.5-coder:7b", #"qwen2.5:latest",
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                code = response.json().get("response", "").strip()
                # Clean up code if it contains markdown formatting
                code = re.sub(r'```python|```', '', code).strip()
                # Remove any print() statements
                code = re.sub(r'print\s*\(([^)]*)\)', r'\1', code)
                return code, None
            else:
                return None, f"Error communicating with FinanceLLM: {response.status_code}"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def execute_query(self, code):
        """Execute the pandas query code"""
        if self.df is None:
            return None, "No CSV file loaded yet."
        
        # Create a safe local namespace with only the dataframe and necessary modules
        local_namespace = {
            "df": self.df, 
            "pd": pd,
            "np": __import__('numpy')
        }
        
        try:
            # Execute the code
            result = eval(code, {"__builtins__": {}}, local_namespace)
            return result, None
        except Exception as e:
            try:
                # If eval fails, try exec for operations that don't return values
                # Add a result variable to capture the final output
                modified_code = code + "\n"
                # Look for the last expression in the code and assign it to result if possible
                lines = code.strip().split('\n')
                if lines:
                    last_line = lines[-1]
                    # If the last line is an expression (not an assignment or other statement)
                    if not re.match(r'^[a-zA-Z0-9_]+ *=', last_line) and not last_line.startswith(('if ', 'for ', 'while ', 'def ')):
                        modified_code = '\n'.join(lines[:-1]) + "\nresult = " + last_line
                    else:
                        modified_code = code + "\nresult = df"  # Default to returning the dataframe
                
                exec(modified_code, {"__builtins__": {}}, local_namespace)
                # Try to get the result which might be stored in a variable
                result = local_namespace.get("result", "Operation completed successfully but returned no result.")
                return result, None
            except Exception as e2:
                return None, f"Error executing query: {str(e2)}"
    
    def analyze_result(self, result, prompt):
        """Analyze the result and provide insights using FinanceLLM"""
        if isinstance(result, pd.DataFrame):
            result_data = result.head(10).to_dict(orient='records')
            result_type = "DataFrame"
            result_shape = result.shape
        elif isinstance(result, pd.Series):
            result_data = result.head(10).to_dict()
            result_type = "Series"
            result_shape = result.shape
        else:
            result_data = str(result)
            result_type = type(result).__name__
            result_shape = None
        
        system_prompt = (
            "You are a data analysis assistant. Given a user question and the result of a pandas query, "
            "provide a brief analysis of the results. Focus on key insights, patterns, and answering "
            "the user's original question. Keep your analysis concise but informative."
        )
        
        user_prompt = f"""
User question: {prompt}

Query result type: {result_type}
Result shape (if applicable): {result_shape}
Result data: {json.dumps(result_data, default=str, indent=2)}

Provide a brief analysis of these results as they relate to the user's question.
"""
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "qwen2.5-coder:7b", #"qwen2.5:latest",
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                analysis = response.json().get("response", "").strip()
                return analysis
            else:
                return f"Could not analyze results: {response.status_code}"
        except Exception as e:
            return f"Error analyzing results: {str(e)}"

def create_visualization(result):
    """Create appropriate visualization based on the data"""
    if not isinstance(result, (pd.DataFrame, pd.Series)):
        return None
    
    plt.figure(figsize=(10, 6))
    
    if isinstance(result, pd.Series):
        if result.dtype in ['int64', 'float64']:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(result, ax=ax)
            ax.set_title(f'Distribution of {result.name}')
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            result.value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Counts of {result.name}')
            return fig
    
    elif isinstance(result, pd.DataFrame):
        if len(result) > 0 and len(result.columns) > 0:
            # For simple 2D data with numeric columns
            numeric_cols = result.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numeric_cols) >= 2 and len(result) <= 1000:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=result, x=numeric_cols[0], y=numeric_cols[1], ax=ax)
                ax.set_title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
                return fig
            elif len(numeric_cols) >= 1 and len(result) <= 1000:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=result, x=numeric_cols[0], ax=ax)
                ax.set_title(f'Distribution of {numeric_cols[0]}')
                return fig
            
    return None

def should_show_visualization(query):
    """Determine if visualization should be shown based on user query"""
    visualization_keywords = [
        "visualize", "visualization", "visualise", "visualisation",
        "plot", "chart", "graph", "display", "show", "draw",
        "histogram", "bar chart", "scatter plot", "line graph"
    ]
    
    query_lower = query.lower()
    
    # Check for explicit visualization requests
    for keyword in visualization_keywords:
        if keyword in query_lower:
            return True
            
    return False

# Set up Streamlit interface
st.set_page_config(page_title="FinanceLLM ChatBot", layout="wide")
st.title("FinanceLLM ChatBot")

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = CSVChatBot()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I'm your CSV analysis assistant. Please upload a CSV file to get started."},
        {"role": "user", "content": "Can you show me the top 5 rows of the data?"},
        {"role": "assistant", "content": "I need you to upload a CSV file first. Once you've done that, I can show you the top rows."},
        {"role": "user", "content": "What columns have the most missing values?"},
        {"role": "assistant", "content": "Please upload a CSV file first. After that, I can analyze the missing values for you."}
    ]
if 'code_history' not in st.session_state:
    st.session_state.code_history = []

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file and st.sidebar.button("Load CSV"):
    result = st.session_state.chatbot.load_csv(uploaded_file)
    st.session_state.chat_history.append({"role": "assistant", "content": result})

# Display DataFrame info if loaded
if st.session_state.chatbot.df is not None:
    with st.sidebar.expander("Data Information"):
        df_info = st.session_state.chatbot.get_df_info()
        st.write(f"Rows: {df_info['shape'][0]}, Columns: {df_info['shape'][1]}")
        st.write("Columns:")
        for col, dtype in df_info['dtypes'].items():
            st.write(f"- {col} ({dtype})")
    
    with st.sidebar.expander("Sample Data"):
        st.dataframe(st.session_state.chatbot.df.head())

# Add visualization toggle in sidebar
st.sidebar.subheader("Settings")
always_visualize = st.sidebar.checkbox("Always show visualizations", value=False)

# Display chat
st.subheader("Chat")
chat_container = st.container()

with chat_container:
    df_results = st.session_state.chatbot.setup_analyzer()
    st.write(df_results)
    st.markdown(f"Account Balance: {df_results['balance_stats']['mean']:.2f} (AVG), Min: {df_results['balance_stats']['min']} , Max: {df_results['balance_stats']['max']}")
    st.write(f"Demographics Industry: {df_results['demographics']['industry_distribution']}")
    # st.write("Columns:")
    # for col, dtype in df_results['dtypes'].items():
    #     st.write(f"- {col} ({dtype})")

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**USER**: {message['content']}")
        else:
            st.markdown(f"**CSV_Assistant**: {message['content']}")

# Input for new question
with st.form(key="query_form"):
    user_query = st.text_area("Ask a question about your data:", key="user_query")
    show_code = st.checkbox("Show generated pandas code")
    show_viz = st.checkbox("Show visualization for this query", value=False)
    submit_button = st.form_submit_button("Ask")

    if submit_button and user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Get a thinking message placeholder
        with st.spinner("Processing..."):
            # Convert prompt to pandas query
            code, error = st.session_state.chatbot.prompt_to_query(user_query)
            
            if error:
                st.session_state.chat_history.append({"role": "assistant", "content": error})
            else:
                # Store code in history
                st.session_state.code_history.append(code)
                st.code(st.session_state.code_history[-1], language="python")
                
                # Execute the query
                result, error = st.session_state.chatbot.execute_query(code)
                
                if error:
                    st.session_state.chat_history.append({"role": "assistant", "content": error})
                else:
                    # Analyze the result
                    analysis = st.session_state.chatbot.analyze_result(result, user_query)
                    
                    # Format the response
                    response = analysis
                    
                    if isinstance(result, pd.DataFrame):
                        if len(result) > 10:
                            result_str = f"\n\nFirst 10 rows of result:\n```\n{result.head(10).to_string()}\n```"
                        else:
                            result_str = f"\n\nResult:\n```\n{result.to_string()}\n```"
                        response += result_str
                    elif isinstance(result, pd.Series):
                        if len(result) > 10:
                            result_str = f"\n\nFirst 10 values of result:\n```\n{result.head(10).to_string()}\n```"
                        else:
                            result_str = f"\n\nResult:\n```\n{result.to_string()}\n```"
                        response += result_str
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Check if visualization should be shown
                    should_viz = (always_visualize or 
                                 show_viz or 
                                 should_show_visualization(user_query))
                    
                    # Create visualization if requested
                    if should_viz:
                        viz = create_visualization(result)
                        if viz:
                            st.subheader("Visualization")
                            st.pyplot(viz)
                        else:
                            st.info("No suitable visualization could be created for this result.")

# Code history display
if show_code and st.session_state.code_history:
    st.subheader("Code equivalent")
    st.code(st.session_state.code_history[-1], language="python")