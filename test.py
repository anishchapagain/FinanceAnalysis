import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Banking Analytics Q&A",
    page_icon="🏦",
    layout="wide"
)

# Function to load sample data
@st.cache_data
def load_sample_data():
    # Sample data as provided
    data = [
        ['284', 'NEPAL RASTRA BANK', 'NP', 'NP', 'FINANCE & FIN. INST.', 'SSECTOR', 'Head Office', 'Nostro', '014500002849', 'NPR', 'Nostro A/c - NRB', 20000.0, 20000.0, False, False, True, False, False, '', '19-Dec-16', '11-Jan-25', '19-Dec-16', '19-Dec-01'],
        ['1012', 'SUM BAHADUR TAMANG', 'NP', 'NP', 'STAFF', 'LOCAL - PERSONS', 'Head Office', 'AACCOUNT', '011800010129', 'NPR', 'CCATEGORY', 250000.0, 250000.0, True, False, True, True, True, '9841119237', '30-Aug-17', '01-Sep-17', '04-Oct-17', '19-Oct-32'],
        ['766', 'BEDVYAS NIRMAN SEWA', 'NP', 'NP', 'IINDUSTRY', 'PROPRIETORY CONCERN', 'Head Office', 'All Other Types', '019900007668', 'NPR', 'Margin on Guarantee', 0.0, 0.0, False, False, False, True, True, '20010321', '30-Jan-02', '12-Aug-01', '23-Oct-45', '19-Dec-01'],
        ['68872', 'JITENDRA SUNCHAURI', 'NP', 'NP', 'STAFF', 'LOCAL - PERSONS', 'Head Office', 'Savings account - staff', '012100688725', 'NPR', 'Savings Account (STAFF)', 39208.34, 39208.34, True, True, True, True, False, '9816668513', '12-Sep-17', '03-Jan-25', '04-Jan-25', '27-Aug-46'],
        ['101775', 'DURGA KARKI', 'NP', 'NP', 'STAFF', 'LOCAL - PERSONS', 'Head Office', 'Savings account - staff', '012101017751', 'NPR', 'Savings Account (STAFF)', 37848.46, 37848.46, False, False, True, True, False, '9804102980', '28-Apr-09', '11-Jan-25', '18-Apr-19', '29-May-38'],
        ['288', 'NEPAL BANK LIMITED', 'NP', 'NP', 'COMMERCIAL BANKS A CLASS FI', 'PUBLIC LIMITED COMPANY', 'Head Office', 'Nostro', '014500002887', 'NPR', 'Nostro A/c - Local Banks', 17798646.0, 17798646.0, False, False, False, False, False, '20190421', '01-Apr-19', '19-Dec-16', '06-Jan-25', '19-Dec-01'],
        ['10722', 'SHARMILA GAUTAM', 'NP', 'NP', 'INDIVIDUALS', 'LOCAL - PERSONS', 'Damauli', 'Savings regular & convertible', '122200107224', 'NPR', 'Savings Account(DAM)', 1045.23, 1045.23, False, False, True, True, True, '9807951381', '23-May-06', '10-Oct-10', '01-Jan-07', '07-Apr-19']
    ]
    
    columns = ['customer_id', 'customer_name', 'nationality', 'residency_status', 
               'customer_industry', 'economic_sector', 'bank_branch', 'account_type',
               'account_number', 'currency_code', 'account_category', 'account_balance',
               'local_currency_balance', 'mobile_banking', 'internet_banking', 
               'account_service', 'kyc_status', 'account_inactive', 'mobile_number',
               'account_open_date', 'last_debit_date', 'last_credit_date', 'date_of_birth']
    
    df = pd.DataFrame(data, columns=columns)
    
    # Convert date columns to datetime
    date_columns = ['account_open_date', 'last_debit_date', 'last_credit_date', 'date_of_birth']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%d-%b-%y')
    
    return df

# Function to execute predefined queries
def execute_query(df, query_type):
    queries = {
        "Total Balance": {
            "query": "df['account_balance'].sum()",
            "result": lambda df: f"NPR {df['account_balance'].sum():,.2f}"
        },
        "Digital Adoption Rate": {
            "query": "(df['mobile_banking'] | df['internet_banking']).mean() * 100",
            "result": lambda df: f"{((df['mobile_banking'] | df['internet_banking']).mean() * 100):.2f}%"
        },
        "Account Type Distribution": {
            "query": "df['account_type'].value_counts()",
            "result": lambda df: df['account_type'].value_counts(),
            "plot": lambda df: px.pie(names=df['account_type'].value_counts().index, 
                                    values=df['account_type'].value_counts().values,
                                    title="Account Type Distribution")
        },
        "Customer Industry Analysis": {
            "query": "df.groupby('customer_industry')['account_balance'].sum()",
            "result": lambda df: df.groupby('customer_industry')['account_balance'].sum(),
            "plot": lambda df: px.bar(
                df.groupby('customer_industry')['account_balance'].sum().reset_index(),
                x='customer_industry',
                y='account_balance',
                title="Total Balance by Customer Industry"
            )
        },
        "Digital Banking Usage": {
            "query": """df.groupby('customer_industry').apply(
                lambda x: ((x['mobile_banking'] | x['internet_banking']).mean() * 100)
            )""",
            "result": lambda df: df.groupby('customer_industry').apply(
                lambda x: f"{((x['mobile_banking'] | x['internet_banking']).mean() * 100):.2f}%"
            )
        }
    }
    
    if query_type in queries:
        result = queries[query_type]["result"](df)
        plot = queries[query_type].get("plot", None)
        if plot:
            return result, plot(df)
        return result, None
    return None, None

# Main app
def main():
    st.title("🏦 Banking Analytics Q&A Dashboard")
    
    # Load data
    df = load_sample_data()
    
    # Sidebar
    st.sidebar.header("Analysis Options")
    
    # Query selection
    query_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Total Balance", "Digital Adoption Rate", "Account Type Distribution",
         "Customer Industry Analysis", "Digital Banking Usage"]
    )
    
    # Custom query input
    custom_query = st.sidebar.text_area("Or enter a custom question:")
    
    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Data Overview")
        st.dataframe(df.head(), use_container_width=True)
        
    with col2:
        st.subheader("Analysis Results")
        if custom_query:
            st.write("Custom query not implemented in this demo")
        else:
            result, plot = execute_query(df, query_type)
            
            if result is not None:
                st.write("Query:", query_type)
                st.write("Result:", result)
                
                if plot is not None:
                    st.plotly_chart(plot, use_container_width=True)
    
    # Additional metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Accounts", len(df))
    with col2:
        st.metric("Active Accounts", len(df[~df['account_inactive']]))
    with col3:
        st.metric("KYC Completed", f"{(df['kyc_status'].mean() * 100):.1f}%")
    with col4:
        st.metric("Digital Banking Users", len(df[df['mobile_banking'] | df['internet_banking']]))

if __name__ == "__main__":
    main()