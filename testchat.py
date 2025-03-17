import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.express as px
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Bank Data Chatbot",
    page_icon="💰",
    layout="wide"
)

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

# Function to display basic info
def show_basic_info(df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Active Accounts", len(df[df['account_inactive'] == False]))
        
    with col2:
        st.metric("Total Balance (NPR)", f"{df['account_balance'].sum():,.2f}")
        st.metric("Inactive Accounts", len(df[df['account_inactive'] == True]))
    
    st.subheader("Account Types")
    account_types = df['account_type'].value_counts().reset_index()
    account_types.columns = ['Account Type', 'Count']
    fig = px.bar(account_types, x='Account Type', y='Count', color='Count')
    st.plotly_chart(fig)

# Function to execute simple pandas queries based on natural language
def execute_query(query, df):
    query = query.lower().strip()
    result = None
    
    # Pattern matching for common queries
    if "show" in query and "inactive" in query:
        result = df[df['account_inactive'] == True]
        
    elif "show" in query and "active" in query:
        result = df[df['account_inactive'] == False]
    
    elif re.search(r"show.*mobile banking", query):
        result = df[df['mobile_banking'] == True]
        
    elif re.search(r"show.*internet banking", query):
        result = df[df['internet_banking'] == True]
    
    elif "balance greater than" in query or "balance more than" in query:
        amount = float(re.search(r'(\d+(?:\.\d+)?)', query).group(1))
        result = df[df['account_balance'] > amount]
        
    elif "balance less than" in query:
        amount = float(re.search(r'(\d+(?:\.\d+)?)', query).group(1))
        result = df[df['account_balance'] < amount]
    
    elif "show customers from" in query:
        branch = None
        for br in df['bank_branch'].unique():
            if br.lower() in query.lower():
                branch = br
                break
        if branch:
            result = df[df['bank_branch'] == branch]
    
    elif "top" in query and "highest balance" in query:
        match = re.search(r'top\s+(\d+)', query)
        n = int(match.group(1)) if match else 5
        result = df.sort_values('account_balance', ascending=False).head(n)
    
    elif re.search(r"accounts opened (in|during|on)", query):
        # Try to extract date patterns
        year_match = re.search(r'\d{4}', query)
        if year_match:
            year = int(year_match.group())
            result = df[df['account_open_date'].dt.year == year]
    
    elif "show" in query and "staff" in query:
        result = df[df['customer_industry'] == 'STAFF']
    
    # Fallback for complex queries
    else:
        st.warning("I couldn't understand that query. Here are some examples you can try:\n" +
                 "- Show inactive accounts\n" +
                 "- Show customers with mobile banking\n" +
                 "- Show accounts with balance greater than 100000\n" +
                 "- Show top 10 customers with highest balance\n" +
                 "- Show customers from Damauli\n" +
                 "- Show staff accounts")
        return None
    
    return result

# Function to visualize based on query
def create_visualization(query, df):
    query = query.lower().strip()
    
    if "plot" in query and "account types" in query:
        st.subheader("Distribution of Account Types")
        fig = px.pie(df, names='account_type')
        st.plotly_chart(fig)
        
    elif "plot" in query and "balances" in query:
        st.subheader("Account Balance Distribution")
        fig = px.histogram(df, x='account_balance', nbins=20, title="Account Balance Distribution")
        fig.update_layout(xaxis_title="Account Balance", yaxis_title="Count")
        st.plotly_chart(fig)
        
    elif "plot" in query and "branches" in query:
        st.subheader("Customers by Branch")
        branch_counts = df['bank_branch'].value_counts().reset_index()
        branch_counts.columns = ['Branch', 'Count']
        fig = px.bar(branch_counts, x='Branch', y='Count', color='Branch')
        st.plotly_chart(fig)
    
    else:
        st.info("Visualization request not understood. Try: 'Plot account types', 'Plot balances', or 'Plot branches'")

# Main app
def main():
    st.title("📊 Bank Data Chatbot")
    
    # Load the data
    with st.spinner("Loading data..."):
        df = load_data()
        
    # Sidebar for options
    st.sidebar.title("Options")
    option = st.sidebar.radio(
        "Choose an option:",
        ["Dashboard Overview", "Query Data", "Visualize Data"]
    )
    
    # Show data sample in sidebar
    with st.sidebar.expander("View Sample Data"):
        st.dataframe(df.head(5))
    
    # Main content based on selected option
    if option == "Dashboard Overview":
        st.header("Bank Data Dashboard")
        show_basic_info(df)
        
        # Show some key metrics
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mobile_users = len(df[df['mobile_banking'] == True])
            st.metric("Mobile Banking Users", mobile_users, f"{mobile_users/len(df)*100:.1f}%")
            
        with col2:
            internet_users = len(df[df['internet_banking'] == True])
            st.metric("Internet Banking Users", internet_users, f"{internet_users/len(df)*100:.1f}%")
            
        with col3:
            avg_balance = df['account_balance'].mean()
            st.metric("Average Account Balance", f"{avg_balance:,.2f}")
        
        # Show distribution of account categories
        st.subheader("Account Categories")
        category_counts = df['account_category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        fig = px.pie(category_counts, values='Count', names='Category', title='Account Categories')
        st.plotly_chart(fig)
        
    elif option == "Query Data":
        st.header("Query the Bank Data")
        st.write("Enter your question in natural language, and I'll try to find the answer.")
        
        query = st.text_input("Your Query", placeholder="e.g., Show inactive accounts")
        
        if query:
            with st.spinner("Processing your query..."):
                result = execute_query(query, df)
                
            if result is not None and not result.empty:
                st.success(f"Found {len(result)} matching records")
                st.dataframe(result)
                
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
                
    elif option == "Visualize Data":
        st.header("Visualize Bank Data")
        st.write("Enter what you want to visualize.")
        
        viz_query = st.text_input("Visualization Request", placeholder="e.g., Plot account types")
        
        if viz_query:
            with st.spinner("Creating visualization..."):
                create_visualization(viz_query, df)
                
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