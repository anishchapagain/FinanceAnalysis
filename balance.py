import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Financial Statement Analysis",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("Financial Statement Analysis Dashboard")
st.write("A comprehensive tool for financial analysis and decision making")

# Function to format a float value as Nepali currency
def format_nepali_currency(amount):
    """Formats a float value as Nepali currency (e.g., 1,65,514.68)."""
    try:
        amount_str = "{:.2f}".format(amount)  # Format to 2 decimal places
        integer_part, decimal_part = amount_str.split('.')

        # Process for Nepali numbering system (lakhs, crores)
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

# Function to format a number as currency with separators (Nepali format)
def format_nepali_currency_new(amount):
    """Formats a number as currency with thousands separators (Indian format) and a dollar sign.
    Handles negative values in two formats: -1234 or (1234).

    Args:
        amount (int or float or str): The amount to format.

    Returns:
        str: The formatted currency string; returns "N/A" if input is not numeric.
    """
    if pd.isna(amount):
        return "$0"
    elif isinstance(amount, (int, float)):
        amount = float(amount) # Ensure it is float
        sign = "-" if amount < 0 else ""
        amount = abs(int(amount)) #remove decimal and get absolute value
        s = str(amount)
        if len(s) > 3:
            first_part = s[-3:]
            remaining_parts = s[:-3]
            formatted_amount = ""
            while len(remaining_parts) > 0:
                formatted_amount = "," + first_part + formatted_amount
                first_part = remaining_parts[-2:]
                remaining_parts = remaining_parts[:-2]
            formatted_amount = remaining_parts + formatted_amount
            return f"{sign}{formatted_amount}"
        else:
            return f"{sign}{s}"
    elif isinstance(amount, str):
        # Remove parentheses and check for negative sign
        amount_str = amount.replace("(", "").replace(")", "")
        sign = "-" if "(" in amount or amount_str.startswith("-") else ""
        #try converting to float
        try:
            amount_num = float(amount_str)
            amount_num = abs(int(amount_num))
            s = str(amount_num)
            if len(s) > 3:
                first_part = s[-3:]
                remaining_parts = s[:-3]
                formatted_amount = ""
                while len(remaining_parts) > 0:
                    formatted_amount = "," + first_part + formatted_amount
                    first_part = remaining_parts[-2:]
                    remaining_parts = remaining_parts[:-2]
                formatted_amount = remaining_parts + formatted_amount
                return f"{sign}{formatted_amount}"
            else:
                return f"{sign}{s}"

        except ValueError:
            return "N/A"
    else:
        return "N/A"

# Function to create a download link for the dataframe
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href


def calculate_financial_metrics(df):
        """
        Calculate financial metrics from the provided dataframe.
        Metrices are calculate from the balance sheet and income statement data.
        """
        metrics = {}
        # Convert dataframe to dict for easier access, since we have only one row
        data = df.iloc[0].to_dict()
        
        # Existing Fields:
        # Total Non-Current Assets + Total Current Assets = Total Assets
        # Total Current Liabilities + Non-Current Liabilities = Total Liabilities
        # Total Liabilities & Equity = Total Liability + Total Equity

        
        print(f"1-Net Income: {data['Net Income']}, Sales Revenue: {data['Sales Revenue']}")
        # Calculate Total Assets
        metrics["Total Current Assets"] = data["Cash"] + data["Accounts Receivable"] + data["Inventory"] 
        metrics["Total Fixed Assets"] = data["Property Plant & Equipment"] + data["Long-term Investments"] # Total Non-Current Assets
        metrics["Total Assets"] = metrics["Total Current Assets"] + metrics["Total Fixed Assets"]
        
        # Calculate Total Liabilities
        metrics["Total Current Liabilities"] = data["Accounts Payable"] + data["Short-term Debt"]
        metrics["Total Long-Term Liabilities"] = data["Long-term Debt"]
        metrics["Total Liabilities"] = metrics["Total Current Liabilities"] + metrics["Total Long-Term Liabilities"]
        
        # Calculate Total Equity
        metrics["Total Equity"] = data["Common Stock"] + data["Retained Earnings"]
        
        # Calculate Depreciation (estimated as 5% of PP&E for this example)
        metrics["Depreciation"] = data["Property Plant & Equipment"] * 0.05 # TODO : Depreciation : 5% of PP&E
        
        # Calculate Amortization (estimated as 2% of Long-term Investments for this example)
        metrics["Amortization"] = data["Long-term Investments"] * 0.02 # TODO : Amortization : 2% of Long-term Investments
        
        # Income Statement
        # operating_income = data["Gross Profit"] - data["total_operating_expenses"] # TODO : Operating Income : Gross Profit - Total Operating Expenses
        
        # Calculate EBITDA: Net Profit + Depreciation (abs) + Amortization (abs)   ---- # 1:  (+ve : Green, -ve : Red)
        # 1
        metrics["EBITDA"] = data["Operating Income"] + metrics["Depreciation"] + metrics["Amortization"] # TODO : EBITDA : Operating Income + Depreciation + Amortization
        
        # Calculate Net Tangible Assets
        metrics["NTA"] = metrics["Total Assets"] - metrics["Total Liabilities"]
        
        # Required financial ratios
        # Debt-to-Equity Ratio (D/E) # 2 (Good) is good : Green, 2-3 (Moderate) is ok : Amber, > 3 (High) is bad : Red
        # DE Ratio (Total Liabilities / Total Equity): Leveraged companies have higher DE ratio
        metrics["DE Ratio"] = metrics["Total Liabilities"] / metrics["Total Equity"]   # Leverage Ratio (MAX: )
        
        # Gear Ratio (Total Debt / Total Equity) # 3: (+ve : Green, -ve : Red)               (Total Equity | Net Worth) = Total Bank Debt / Total Equity (Max: 4:1)
        metrics["Gear Ratio"] = (data["Short-term Debt"] + data["Long-term Debt"]) / metrics["Total Equity"]
        
        # Liquidity Ratio (Current Assets / Current Liabilities)
        metrics["Liquidity Ratio"] = metrics["Total Current Assets"] / metrics["Total Current Liabilities"]
        
        # Leverage Ratio (Total Assets / Total Equity)
        metrics["Leverage Ratio"] = metrics["Total Assets"] / metrics["Total Equity"]
        
        # Current Ratio (Current Assets / Current Liabilities)
        # 6
        # Strong: 2.0 or red, > 1.5: Green, Adequate: 1.5, Weak: 1.0 or lower 
        metrics["Current Ratio"] = metrics["Total Current Assets"] / metrics["Total Current Liabilities"] # 1.5:1 (Good) is good : Green, 1:1 (Moderate) is ok : Amber, < 1:1 (Weak) is bad : Red
        
        # Quick Ratio (Cash + Accounts Receivable) / Current Liabilities
        # 7 : current asset - stock # inventory
        # Above 1.0 is good : Green, below:1 > Red                                      # 0.7 to 1.0 is ok : Amber, < 0.7 is bad : Red
        metrics["Quick Ratio"] = (data["Cash"] + data["Accounts Receivable"]) / metrics["Total Current Liabilities"]  # Current Asset - Inventory|stock|work in progress
        

        # Net Profit Margin (Net Income / Sales Revenue) # 2:  (+ve : Green, -ve : Red) -- Income Statement
        # ROA = Net Income / Total Assets # 3:  (+ve : Green, -ve : Red) -- Balance Sheet & Income Statement

        # Return on Assets (ROA) (Net Income / Total Assets)
        metrics["ROA"] = data["Net Income"] / metrics["Total Assets"] # Balance Sheet & Income Statement
        # print(f"ROA {metrics['ROA']} = {data['Net Income']} - {metrics['Total Assets']}")
        
        # Interest Coverage Ratio (EBIT / Interest Expense) # 4 (Strong) is good : Green, 3-4 (Moderate) is ok : Amber, < 3 (Weak) is bad : Red
        # metrics["Interest Coverage Ratio"] = data["Operating Income"] / data["Interest Expense"]  # TODO : Interest Coverage : - Operating Income / Interest Expense # Revolving around 1.5 to 2.0 is good
        metrics["Interest Coverage Ratio"] = data["Operating Income"] / abs(data["Interest Expense"])  # TODO : Interest Coverage : - Operating Income / Interest Expense
        # Interest Expense: Negative values are converted to positive for calculation (Amber / Red flag)


        # Revolving: Working Capital
        # Product Type: Revolving (ICR | DSCR) SAME all others are to be calculated
        # Term Loan: CR | QR | ICR (not required)
        # Revolving & term loan: ALL Ratio Required

        # Permanent Working Capital

        # Rank the Ratio
        # EBITDA - 1 (if -ve )
        
        # Decision Support: Preliminary Analysis
        # Risk Based Pricing #  TODO
        # Credit Score: 750+ (Green), 700-750 (Amber), < 700 (Red) # TODO

        # Debt Service Coverage Ratio (EBITDA / (Principal Payments + Interest Expense))  # 5:  
        # Assuming Principal Payments is 10% of Long-term Debt for this example
        principal_payments = data["Long-term Debt"] * 0.1 # TODO : Principal Payments : 10% of Long-term Debt
        debt_service = principal_payments + abs(data["Interest Expense"]) # TODO: Debt Service : Principal Payments + Interest Expense
        # dscr = (principal_payments + abs(data["Interest Expense"])) # TODO: DSCR : EBITDA / (Principal Payments + Interest Expense)  # 1 (Strong) is good : Green, 0.9 :Amber,  < 0.9 (Weak) is bad : Red
        if debt_service > 0:
            metrics["DSCR"] = metrics["EBITDA"] / debt_service
        else:
            metrics["DSCR"] = float("inf") if metrics["EBITDA"] > 0 else 0
        
        
        # NTA: Stock Receivables : 
        # metrics["Stock Receivables"] = metrics["NTA"] / data["Accounts Receivable"] # TODO : Stock Receivables : NTA / Accounts Receivable
        
        # Profitability Metrics
        metrics["Gross Margin"] = data["Gross Profit"] / data["Sales Revenue"]
        metrics["Operating Margin"] = data["Operating Income"] / data["Sales Revenue"]
        metrics["Net Profit Margin"] = data["Net Income"] / data["Sales Revenue"] # Income Statement (0.1667)
        metrics["Return on Equity"] = data["Net Income"] / metrics["Total Equity"]
        
        # Efficiency Metrics
        metrics["Asset Turnover"] = data["Sales Revenue"] / metrics["Total Assets"]
        metrics["Inventory Turnover"] = data["Cost of Goods Sold"] / data["Inventory"]
        metrics["Receivables Turnover"] = data["Sales Revenue"] / data["Accounts Receivable"]
        
        # Cash Flow Metrics
        metrics["Operating Cash Flow Ratio"] = data["Net Cash from Operations"] / metrics["Total Current Liabilities"] # TODO Cash Flow
        metrics["Cash Flow to Debt Ratio"] = data["Net Cash from Operations"] / metrics["Total Liabilities"] # TODO Cash Flow
        
        # Negative values for Cash Flow to Debt Ratio are not possible, so we set it to 0 if negative
        # - COGS

        # Convert to dataframe for display
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()})
        print(f"1-Net Income: {data['Net Income']}, Sales Revenue: {data['Sales Revenue']}")
        print(f"1-Metrices DF: {metrics_df.to_json()}")
        return metrics_df


def format_ratio(ratio_name, value):
            """
            Format the financial ratio value with appropriate labels and colors.
            """
            if ratio_name == "DE Ratio": # DE Ratio - Leverage Ratio
                if value <= 1.5:
                    return f"{value:.2f} (Good)"
                elif value <= 2.0:
                    return f"{value:.2f} (Moderate)"
                else:
                    return f"{value:.2f} (High)"
            elif ratio_name == "Current Ratio" or ratio_name == "Liquidity Ratio":
                if value >= 2.0:
                    return f"{value:.2f} (Strong)"
                elif value >= 1.0:
                    return f"{value:.2f} (Adequate)"
                else:
                    return f"{value:.2f} (Weak)"
            elif ratio_name == "Quick Ratio":
                if value >= 1.0:
                    return f"{value:.2f} (Strong)"
                elif value >= 0.7:
                    return f"{value:.2f} (Adequate)"
                else:
                    return f"{value:.2f} (Weak)"
            elif ratio_name in ["ROA", "Return on Equity"]:
                if value >= 0.1:
                    return f"{value:.2%} (Good)"
                elif value >= 0.05:
                    return f"{value:.2%} (Moderate)"
                else:
                    return f"{value:.2%} (Low)"
            elif ratio_name == "Interest Coverage Ratio":
                if value >= 3.0:
                    return f"{value:.2f} (Strong)"
                elif value >= 1.5:
                    return f"{value:.2f} (Adequate)"
                else:
                    return f"{value:.2f} (Weak)"
            elif ratio_name == "DSCR":
                if value >= 1.5:
                    return f"{value:.2f} (Strong)"
                elif value >= 1.0:
                    return f"{value:.2f} (Adequate)"
                else:
                    return f"{value:.2f} (Weak)"
            else:
                return f"{value:.2f}"
            

# Sidebar
st.sidebar.title("Upload Financial Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Create sample data button
if st.sidebar.button("Use Sample Data"):
    # Create sample data based on the structure from balance.csv
    sample_data = {
        "Cash": [500000],
        "Accounts Receivable": [1200000],
        "Inventory": [3000000],
        "Property Plant & Equipment": [10000000],
        "Long-term Investments": [0],
        "Accounts Payable": [3000000],
        "Short-term Debt": [2000000],
        "Long-term Debt": [4000000],
        "Common Stock": [2000000],
        "Retained Earnings": [3700000],
        "Sales Revenue": [15000000],
        "Cost of Goods Sold": [-12000000],
        "Gross Profit": [1000000],
        "Research & Development": [0],
        "Marketing": [0],
        "Administrative Costs": [0],
        "Operating Income": [0],
        "Interest Expense": [0],
        "Income Before Tax": [0],
        "Tax Expense_20": [0],
        "Net Income": [0],
        "Net Cash from Operations": [-3500000],
        "Net Cash from Investing": [-500000],
        "Net Cash from Financing": [-1000000],
        "Net Change in Cash": [-5000000]
    }
    df = pd.DataFrame(sample_data)
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # # Apply the formatting function to all numeric columns
    # for col in numeric_cols:
    #     df[col] = df[col].apply(format_nepali_currency)
    
    # Convert to CSV and create a BytesIO object
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Set the uploaded_file to the sample data
    uploaded_file = csv_buffer

# Main content
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Apply the formatting function to all numeric columns
    # for col in numeric_cols:
    #     df[col] = df[col].apply(format_nepali_currency)
    
    # Display raw data
    with st.expander("Raw Financial Data", expanded=False):
        st.dataframe(df)
        st.markdown(get_table_download_link(df, "financial_data", "Download CSV"), unsafe_allow_html=True)
    
    # Calculate all metrics
    metrics_df = calculate_financial_metrics(df) # 1
    
    # Display metrics
    ratios = {}

    ebitda = metrics_df["EBITDA"].iloc[0]
    icr = format_ratio("Interest Coverage Ratio", metrics_df["Interest Coverage Ratio"].iloc[0])
    dscr = format_ratio("DSCR", metrics_df["DSCR"].iloc[0])
    gear_ratio = format_ratio("Gear Ratio", metrics_df["Gear Ratio"].iloc[0])
    leverage_ratio = format_ratio("Leverage Ratio", metrics_df["Leverage Ratio"].iloc[0])
    current_ratio = format_ratio("Current Ratio", metrics_df["Current Ratio"].iloc[0])
    quick_ratio = format_ratio("Quick Ratio", metrics_df["Quick Ratio"].iloc[0])
    liquidity_ratio = format_ratio("Liquidity Ratio", metrics_df["Liquidity Ratio"].iloc[0])
    debt_equity_ratio = format_ratio("DE Ratio", metrics_df["DE Ratio"].iloc[0])

    success = 0
    # total asset == total liability

    if ebitda < 0:
        st.error("EBITDA is negative, indicating potential financial distress.")
    elif ebitda == 0:
        st.warning("EBITDA is zero, indicating no operational profit.")
    elif ebitda > 0:
        st.success(f"**EBITDA {ebitda} is positive**, indicating healthy operational performance.")
        success += 1
    else:
        st.info("EBITDA is not available.")

    if ebitda > 0:
        if "weak" in icr.lower():
            st.error(f"Interest Coverage Ratio  {icr}, indicating potential difficulty in meeting interest payments.")
        elif "adequate" in icr.lower():
            st.warning(f"Interest Coverage Ratio {icr}, indicating potential risk.")
        elif "strong" in icr.lower():
            st.success(f"**Interest Coverage Ratio: {icr}**, indicating good ability to meet interest payments.")
            success += 1
        else:
            st.error("Interest Coverage Ratio is not available.")


        if "weak" in dscr.lower():
            st.error(f"Debt Service Coverage Ratio {dscr}, indicating potential difficulty in meeting debt obligations.")   
        elif "adequate" in dscr.lower():
            st.warning(f"Debt Service Coverage Ratio {dscr}, indicating potential risk.")
        elif "strong" in dscr.lower():
            st.success(f"**Debt Service Coverage Ratio: {dscr}**, indicating good ability to meet debt obligations.")
            success += 1
        else:
            st.error("Debt Service Coverage Ratio is not available.")

        
        if "weak" in gear_ratio.lower():
            st.error(f"Gear Ratio {gear_ratio}, indicating high leverage.")
        elif "moderate" in gear_ratio.lower():
            st.warning(f"Gear Ratio {gear_ratio}, indicating moderate leverage.")
        elif "good" in gear_ratio.lower():
            st.success(f"**Gear Ratio: {gear_ratio}**, indicating good leverage.")
            success += 1
        else:
            st.error(f"Gear Ratio is not available. {gear_ratio}")


        if "weak" in leverage_ratio.lower():
            st.error(f"Leverage Ratio {leverage_ratio}, indicating high leverage.")
        elif "moderate" in leverage_ratio.lower():
            st.warning(f"Leverage Ratio {leverage_ratio}, indicating moderate leverage.")
        elif "good" in leverage_ratio.lower():
            st.success(f"**Leverage Ratio: {leverage_ratio}**, indicating good leverage.")
            success += 1
        else:
            st.error(f"Leverage Ratio is not available. {leverage_ratio}")


        if "weak" in current_ratio.lower():
            st.error(f"Current Ratio {current_ratio}, indicating potential liquidity issues.")
        elif "adequate" in current_ratio.lower():
            st.warning(f"Current Ratio {current_ratio}, indicating potential risk.")
        elif "strong" in current_ratio.lower():
            st.success(f"**Current Ratio: {current_ratio}**, indicating good liquidity.")
            success += 1
        else:
            st.error("Current Ratio is not available.")

        
        if "weak" in quick_ratio.lower():
            st.error(f"Quick Ratio {quick_ratio}, indicating potential liquidity issues.")
        elif "adequate" in quick_ratio.lower():
            st.warning(f"Quick Ratio {quick_ratio}, indicating potential risk.")
        elif "strong" in quick_ratio.lower():
            st.success(f"**Quick Ratio: {quick_ratio}**, indicating good liquidity.")
            success += 1
        else:
            st.error("Quick Ratio is not available.")
        

    if success > 3:
        st.write("**Overall Financial Health: Good**")
    elif success == 3:
        st.write("**Overall Financial Health: Moderate**")    
    elif success < 3:
        st.write("**Overall Financial Health: Weak**")
    st.divider()
    if "weak" in liquidity_ratio.lower():
        st.error(f"Liquidity Ratio {liquidity_ratio}, indicating potential liquidity issues.")
    elif "adequate" in liquidity_ratio.lower():
        st.warning(f"Liquidity Ratio {liquidity_ratio}, indicating potential risk.")
    elif "strong" in liquidity_ratio.lower():
        st.success(f"**Liquidity Ratio: {liquidity_ratio}**, indicating good liquidity.")
    else:
        st.error("Liquidity Ratio is not available.")

    
    if "weak" in debt_equity_ratio.lower():
        st.error(f"Debt to Equity Ratio {debt_equity_ratio}, indicating high leverage.")
    elif "moderate" in debt_equity_ratio.lower():
        st.warning(f"Debt to Equity Ratio {debt_equity_ratio}, indicating moderate leverage.")
    elif "good" in debt_equity_ratio.lower():
        st.success(f"**Debt to Equity Ratio: {debt_equity_ratio}**, indicating good leverage.")
    else:
        st.error("Debt to Equity Ratio is not available.")


    # st.write("**EBITDA**:", f"${metrics_df['EBITDA'].iloc[0]:,.0f}")
    # st.write(f"**Interest Coverage Ratio**: {format_ratio('Interest Coverage Ratio', metrics_df['Interest Coverage Ratio'].iloc[0])}")
    # st.write(f"**DSCR**: {format_ratio('DSCR', metrics_df['DSCR'].iloc[0])}")
    # st.write(f"**Gear Ratio**: {format_ratio('Gear Ratio', metrics_df['Gear Ratio'].iloc[0])}")
    # st.write(f"**Leverage Ratio**: {format_ratio('Leverage Ratio', metrics_df['Leverage Ratio'].iloc[0])}")
    # st.write(f"**Current Ratio**: {format_ratio('Current Ratio', metrics_df['Current Ratio'].iloc[0])}")
    # st.write(f"**Quick Ratio**: {format_ratio('Quick Ratio', metrics_df['Quick Ratio'].iloc[0])}")
    # st.divider()
    # st.write(f"**D/E Ratio**: {format_ratio('DE Ratio', metrics_df['DE Ratio'].iloc[0])}")
    # st.write(f"**Liquidity Ratio**: {format_ratio('Liquidity Ratio', metrics_df['Liquidity Ratio'].iloc[0])}")

    
    
    # Create columns for the dashboard
    col1, col2 = st.columns(2)
    
    # Key Financial Summary Section
    with col1:
        st.header("Key Financial Summary")
        
        # Summary cards for key figures
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.metric("Total Assets", f"${metrics_df['Total Assets'].iloc[0]:,.0f}")
            st.metric("Total Liabilities", f"${metrics_df['Total Liabilities'].iloc[0]:,.0f}")
            st.metric("Total Equity", f"${metrics_df['Total Equity'].iloc[0]:,.0f}")
        
        with summary_col2:
            st.metric("Sales Revenue", f"${df['Sales Revenue'].iloc[0]:,.0f}")
            st.metric("Net Income", f"${df['Net Income'].iloc[0]:,.0f}")
            st.metric("EBITDA", f"${metrics_df['EBITDA'].iloc[0]:,.0f}")
        
        # Balance Sheet Structure Visualization
        st.subheader("Balance Sheet Structure")
        
        # Create data for the pie charts
        assets_data = [
            metrics_df["Total Current Assets"].iloc[0],
            metrics_df["Total Fixed Assets"].iloc[0]
        ]
        assets_labels = ["Current Assets", "Fixed Assets"]
        
        liab_equity_data = [
            metrics_df["Total Current Liabilities"].iloc[0],
            metrics_df["Total Long-Term Liabilities"].iloc[0],
            metrics_df["Total Equity"].iloc[0]
        ]
        liab_equity_labels = ["Current Liabilities", "Long-Term Liabilities", "Equity"]
        
        # Create the figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "pie"}]],
            subplot_titles=("Assets Structure", "Liabilities & Equity Structure")
        )
        
        # Add the pie traces
        fig.add_trace(
            go.Pie(
                labels=assets_labels,
                values=assets_data,
                textinfo="label+percent",
                hole=0.3
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Pie(
                labels=liab_equity_labels,
                values=liab_equity_data,
                textinfo="label+percent",
                hole=0.3
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=500, margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Financial Ratios Section
    with col2:
        st.header("Financial Ratios Analysis: EBITDA | DE Ratio | Gear | Interest Coverage Ratio | DSCR | CR | QR")
        
        # Create columns for the financial ratios
        ratio_col1, ratio_col2 = st.columns(2)
        
        # Function to format ratios with color indicators
        
        
        # Display key financial ratios
        with ratio_col1:
            st.subheader("Solvency & Leverage")
            st.write(f"**D/E Ratio**: {format_ratio('DE Ratio', metrics_df['DE Ratio'].iloc[0])}")
            st.write(f"**Gear Ratio**: {format_ratio('Gear Ratio', metrics_df['Gear Ratio'].iloc[0])}")
            st.write(f"**Leverage Ratio**: {format_ratio('Leverage Ratio', metrics_df['Leverage Ratio'].iloc[0])}")
            
            st.subheader("Profitability")
            st.write(f"**ROA**: {format_ratio('ROA', metrics_df['ROA'].iloc[0])}")
            st.write(f"**Return on Equity**: {format_ratio('Return on Equity', metrics_df['Return on Equity'].iloc[0])}")
            st.write(f"**Gross Margin**: {metrics_df['Gross Margin'].iloc[0]:.2%}")
            st.write(f"**Net Profit Margin**: {metrics_df['Net Profit Margin'].iloc[0]:.2%}")
        
        with ratio_col2:
            st.subheader("Liquidity")
            st.write(f"**Current Ratio**: {format_ratio('Current Ratio', metrics_df['Current Ratio'].iloc[0])}")
            st.write(f"**Quick Ratio**: {format_ratio('Quick Ratio', metrics_df['Quick Ratio'].iloc[0])}")
            st.write(f"**Liquidity Ratio**: {format_ratio('Liquidity Ratio', metrics_df['Liquidity Ratio'].iloc[0])}")
            
            st.subheader("Debt Service")
            st.write(f"**Interest Coverage Ratio**: {format_ratio('Interest Coverage Ratio', metrics_df['Interest Coverage Ratio'].iloc[0])}")
            st.write(f"**DSCR**: {format_ratio('DSCR', metrics_df['DSCR'].iloc[0])}")
            st.write(f"**Cash Flow to Debt**: {metrics_df['Cash Flow to Debt Ratio'].iloc[0]:.2f}") # TODO 
        
        # Financial Ratios Radar Chart
        st.subheader("Financial Ratios Radar Chart")
        
        # Prepare data for radar chart
        categories = ['Liquidity', 'Profitability', 'Efficiency', 'Solvency', 'Growth']
        
        # Normalize values between 0 and 1 for the radar chart
        # These are just example mappings - would need to be calibrated for real-world use
        liquidity_score = min(1, metrics_df['Current Ratio'].iloc[0] / 3)
        profitability_score = min(1, metrics_df['ROA'].iloc[0] * 10)
        efficiency_score = min(1, metrics_df['Asset Turnover'].iloc[0])
        solvency_score = min(1, 1 / (metrics_df['DE Ratio'].iloc[0] if metrics_df['DE Ratio'].iloc[0] > 0 else 1))
        
        # For growth, we're using placeholder value since we don't have historical data
        # In a real app, this would come from comparing current to previous periods
        growth_score = 0.65  # Placeholder TODO 
        
        values = [liquidity_score, profitability_score, efficiency_score, solvency_score, growth_score]
        
        # Create the radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Company Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=500,
            margin=dict(t=30, b=30, l=30, r=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Analysis Tabs
    st.header("Detailed Financial Analysis")
    
    tabs = st.tabs(["Balance Sheet", "Income Statement", "Cash Flow", "Financial Health", "Decision Support"])
    
    # Balance Sheet Tab
    with tabs[0]:
        # Create columns for the balance sheet displays
        bs_col1, bs_col2 = st.columns(2)
        
        with bs_col1:
            st.subheader("Assets")
            
            assets_df = pd.DataFrame({
                "Category": ["Current Assets", "Current Assets", "Current Assets", "Fixed Assets", "Fixed Assets"],
                "Item": ["Cash", "Accounts Receivable", "Inventory", "Property Plant & Equipment", "Long-term Investments"],
                "Amount": [
                    df["Cash"].iloc[0],
                    df["Accounts Receivable"].iloc[0],
                    df["Inventory"].iloc[0],
                    df["Property Plant & Equipment"].iloc[0],
                    df["Long-term Investments"].iloc[0]
                ]
            })
            
            # Add totals
            assets_df = pd.concat([
                assets_df,
                pd.DataFrame({
                    "Category": ["Total", "Total", "Total"],
                    "Item": ["Total Current Assets", "Total Fixed Assets", "Total Assets"],
                    "Amount": [
                        metrics_df["Total Current Assets"].iloc[0],
                        metrics_df["Total Fixed Assets"].iloc[0],
                        metrics_df["Total Assets"].iloc[0]
                    ]
                })
            ])
            
            # Display as a formatted table
            st.dataframe(
                assets_df,
                column_config={
                    "Category": st.column_config.TextColumn("Category"),
                    "Item": st.column_config.TextColumn("Item"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%d")
                },
                hide_index=True
            )
            
            # Visualization of Asset Composition
            st.subheader("Asset Composition - Treemap")
            
            # Filter out the totals for the visualization
            assets_viz_df = assets_df[~assets_df["Item"].isin(["Total Current Assets", "Total Fixed Assets", "Total Assets"])]
            
            fig = px.treemap(
                assets_viz_df,
                path=["Category", "Item"],
                values="Amount"
            )
            
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with bs_col2:
            st.subheader("Liabilities & Equity")
            
            liab_equity_df = pd.DataFrame({
                "Category": ["Current Liabilities", "Current Liabilities", "Long-term Liabilities", "Equity", "Equity"],
                "Item": ["Accounts Payable", "Short-term Debt", "Long-term Debt", "Common Stock", "Retained Earnings"],
                "Amount": [
                    df["Accounts Payable"].iloc[0],
                    df["Short-term Debt"].iloc[0],
                    df["Long-term Debt"].iloc[0],
                    df["Common Stock"].iloc[0],
                    df["Retained Earnings"].iloc[0]
                ]
            })
            
            # Add totals
            liab_equity_df = pd.concat([
                liab_equity_df,
                pd.DataFrame({
                    "Category": ["Total", "Total", "Total", "Total"],
                    "Item": ["Total Current Liabilities", "Total Long-Term Liabilities", "Total Liabilities", "Total Equity"],
                    "Amount": [
                        metrics_df["Total Current Liabilities"].iloc[0],
                        metrics_df["Total Long-Term Liabilities"].iloc[0],
                        metrics_df["Total Liabilities"].iloc[0],
                        metrics_df["Total Equity"].iloc[0]
                    ]
                })
            ])
            
            # Display as a formatted table
            st.dataframe(
                liab_equity_df,
                column_config={
                    "Category": st.column_config.TextColumn("Category"),
                    "Item": st.column_config.TextColumn("Item"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%d")
                },
                hide_index=True
            )
            
            # Visualization of Liabilities & Equity Composition
            st.subheader("Liabilities & Equity Composition")
            
            # Filter out the totals for the visualization
            liab_equity_viz_df = liab_equity_df[~liab_equity_df["Item"].isin(["Total Current Liabilities", "Total Long-Term Liabilities", "Total Liabilities", "Total Equity"])]
            
            fig = px.treemap(
                liab_equity_viz_df,
                path=["Category", "Item"],
                values="Amount"
            )
            
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Income Statement Tab
    with tabs[1]:
        # Create columns for the income statement displays
        is_col1, is_col2 = st.columns(2)
        
        with is_col1:
            st.subheader("Income Statement")
            
            income_df = pd.DataFrame({ # TODO: ValueError: All arrays must be of the same length
                "Category": ["Revenue", "Expense", "Profit", "Expense", "Expense", "Expense", "Profit", "Expense", "Profit", "Expense", "Profit"],
                "Item": [
                    "Sales Revenue", 
                    "Cost of Goods Sold", 
                    "Gross Profit", 
                    "Research & Development", 
                    "Marketing", 
                    "Administrative Costs", 
                    "Operating Income", 
                    "Interest Expense", 
                    "Income Before Tax", 
                    "Tax Expense", 
                    "Net Income"
                ],
                "Amount": [
                    df["Sales Revenue"].iloc[0],
                    df["Cost of Goods Sold"].iloc[0],
                    df["Gross Profit"].iloc[0],
                    df["Research & Development"].iloc[0], #
                    df["Marketing"].iloc[0],
                    df["Administrative Costs"].iloc[0],
                    df["Operating Income"].iloc[0],
                    df["Interest Expense"].iloc[0],
                    df["Income Before Tax"].iloc[0],
                    df["Tax Expense_20"].iloc[0],
                    df["Net Income"].iloc[0]
                ]
            })
            
            # Display as a formatted table
            st.dataframe(
                income_df,
                column_config={
                    "Category": st.column_config.TextColumn("Category"),
                    "Item": st.column_config.TextColumn("Item"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%d")
                },
                hide_index=True, use_container_width=True
            )
            
            # EBITDA Calculation
            st.subheader("EBITDA Calculation")
            
            ebitda_df = pd.DataFrame({
                "Item": ["Operating Income", "Depreciation", "Amortization", "EBITDA"],
                "Amount": [
                    df["Operating Income"].iloc[0],
                    metrics_df["Depreciation"].iloc[0],
                    metrics_df["Amortization"].iloc[0],
                    metrics_df["EBITDA"].iloc[0]
                ]
            })
            
            # Display as a formatted table
            st.dataframe(
                ebitda_df,
                column_config={
                    "Item": st.column_config.TextColumn("Item"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%d")
                },
                hide_index=True
            )
        
        with is_col2:
            # Visualization of Income Statement
            st.subheader("Income Statement Waterfall Chart - Gap Analysis")
            
            # Prepare data for waterfall chart
            waterfall_data = [
                {"name": "Sales Revenue", "value": df["Sales Revenue"].iloc[0]},
                {"name": "Cost of Goods Sold", "value": -df["Cost of Goods Sold"].iloc[0]},
                {"name": "Gross Profit", "isTotal": True},
                {"name": "R&D", "value": -df["Research & Development"].iloc[0]}, # TODO # {"name": "R&D", "value": df.get("Research & Development",0.0)}, # TODO
                {"name": "Marketing", "value": -df["Marketing"].iloc[0]},
                {"name": "Admin", "value": -df["Administrative Costs"].iloc[0]},
                {"name": "Operating Income", "isTotal": True},
                {"name": "Interest", "value": abs(-df["Interest Expense"].iloc[0])}, # TODO
                {"name": "Income Before Tax", "isTotal": True}, 
                {"name": "Tax", "value": -df["Tax Expense_20"].iloc[0]},
                {"name": "Net Income", "isTotal": True}
            ]
            
            # Calculate the totals
            running_total = 0
            for item in waterfall_data:
                if "isTotal" not in item:
                    running_total += item["value"]
                else:
                    item["value"] = running_total
            
            # Create lists for the chart
            names = [item["name"] for item in waterfall_data]
            values = [item["value"] for item in waterfall_data]
            
            # Create measure list (relative or total)
            measure = ["relative" if "isTotal" not in item else "total" for item in waterfall_data]
            
            # Create waterfall chart
            fig = go.Figure(go.Waterfall(
                name="Income Statement",
                orientation="v",
                measure=measure,
                x=names,
                textposition="outside",
                text=values,
                y=values,
                connector={"line": {"color": "rgb(63, 63, 63)"}}
            ))
            
            fig.update_layout(
                title="",
                showlegend=False,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Profitability Ratios Bar Chart
            st.subheader("Profitability Ratios")
            
            # Prepare data for bar chart
            profitability_data = {
                "Ratio": ["Gross Margin", "Operating Margin", "Net Profit Margin", "ROA", "ROE"],
                "Value": [
                    metrics_df["Gross Margin"].iloc[0],
                    metrics_df["Operating Margin"].iloc[0],
                    metrics_df["Net Profit Margin"].iloc[0],
                    metrics_df["ROA"].iloc[0],
                    metrics_df["Return on Equity"].iloc[0]
                ]
            }
            
            prof_df = pd.DataFrame(profitability_data)
            
            fig = px.bar(
                prof_df,
                x="Ratio",
                y="Value",
                text_auto='.2%'
            )
            
            fig.update_layout(
                yaxis=dict(tickformat=".1%"),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Cash Flow Tab
    with tabs[2]:
        # Create columns for the cash flow displays
        cf_col1, cf_col2 = st.columns(2)
        
        with cf_col1:
            st.subheader("Cash Flow Statement")
            
            cash_flow_df = pd.DataFrame({
                "Category": ["Operations", "Investing", "Financing", "Total"],
                "Item": [
                    "Net Cash from Operations", 
                    "Net Cash from Investing", 
                    "Net Cash from Financing", 
                    "Net Change in Cash"
                ],
                "Amount": [
                    df.get("Net Cash from Operations",0.0).iloc[0],
                    df.get("Net Cash from Investing",0.0).iloc[0],
                    df.get("Net Cash from Financing",0.0).iloc[0],
                    df.get("Net Change in Cash",0.0).iloc[0],
                ]
            })
            
            # Display as a formatted table
            st.dataframe(
                cash_flow_df,
                column_config={
                    "Category": st.column_config.TextColumn("Category"),
                    "Item": st.column_config.TextColumn("Item"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%d")
                },
                hide_index=True
            )
            
            # Cash Flow Ratios
            st.subheader("Cash Flow Ratios")
            
            cf_ratios_df = pd.DataFrame({
                "Ratio": [
                    "Operating Cash Flow Ratio",
                    "Cash Flow to Debt Ratio"
                ],
                "Value": [
                    metrics_df["Operating Cash Flow Ratio"].iloc[0],
                    metrics_df["Cash Flow to Debt Ratio"].iloc[0]
                ],
                "Description": [
                    "Ability to cover current liabilities with operating cash flow",
                    "Ability to pay off debt with operating cash flow"
                ]
            })
            
            # Display as a formatted table
            st.dataframe(
                cf_ratios_df,
                column_config={
                    "Ratio": st.column_config.TextColumn("Ratio"),
                    "Value": st.column_config.NumberColumn("Value", format="%.2f"),
                    "Description": st.column_config.TextColumn("Description")
                },
                hide_index=True
            )
        
        with cf_col2:
            # Visualization of Cash Flow Components
            st.subheader("Cash Flow Components")
            
            # Prepare data for bar chart
            cf_data = cash_flow_df.iloc[0:3].copy()  # Exclude the total
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=cf_data["Item"],
                y=cf_data["Amount"],
                text=cf_data["Amount"].apply(lambda x: f"${x:,.0f}"), #TODO
                textposition="outside"
            ))
            
            fig.update_layout(
                title="",
                xaxis_title="",
                yaxis_title="Amount ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cash Flow Waterfall Chart
            st.subheader("Cash Flow Analysis")
            
            # Prepare data for waterfall chart
            cf_waterfall_data = [
                {"name": "Starting Cash", "value": df["Cash"].iloc[0] - df.get("Net Change in Cash",0.0).iloc[0]},
                {"name": "Operations", "value": df.get("Net Cash from Operations",0.0).iloc[0]},
                {"name": "Investing", "value": df.get("Net Cash from Investing",0.0).iloc[0]},
                {"name": "Financing", "value": df.get("Net Cash from Financing",0.0).iloc[0]},
                {"name": "Ending Cash", "isTotal": True}
            ]
            
            # Calculate the total
            running_total = cf_waterfall_data[0]["value"]
            for i in range(1, len(cf_waterfall_data)-1):
                running_total += cf_waterfall_data[i]["value"]
            
            cf_waterfall_data[-1]["value"] = running_total
            
            # Create lists for the chart
            names = [item["name"] for item in cf_waterfall_data]
            values = [item["value"] for item in cf_waterfall_data]
            
            # Create measure list (relative or total)
            measure = ["relative" if "isTotal" not in item else "total" for item in cf_waterfall_data]
            
            # Create waterfall chart
            fig = go.Figure(go.Waterfall(
                name="Cash Flow",
                orientation="v",
                measure=measure,
                x=names,
                textposition="outside",
                text=values, # [f"${val:,.0f}" for val in values],
                y=values,
                connector={"line": {"color": "rgb(63, 63, 63)"}}
            ))
            
            fig.update_layout(
                title="",
                showlegend=False,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    # Financial Health Tab
    with tabs[3]:
        st.subheader("Overall Financial Health Assessment")
        
        # Function to calculate overall financial health score
        def calculate_health_score(metrics_df):
            # Create a dictionary with thresholds
            thresholds = {
                'Current Ratio': {'high': 2.0, 'medium': 1.0, 'high_points': 10, 'medium_points': 5},
                'Quick Ratio': {'high': 1.0, 'medium': 0.7, 'high_points': 10, 'medium_points': 5},
                'Operating Cash Flow Ratio': {'high': 1.0, 'medium': 0.5, 'high_points': 10, 'medium_points': 5},
                'DE Ratio': {'low': 1.0, 'medium': 2.0, 'low_points': 10, 'medium_points': 5, 'inverted': True},
                'Interest Coverage Ratio': {'high': 3.0, 'medium': 1.5, 'high_points': 10, 'medium_points': 5},
                'DSCR': {'high': 1.5, 'medium': 1.0, 'high_points': 10, 'medium_points': 5},
                'ROA': {'high': 0.1, 'medium': 0.05, 'high_points': 10, 'medium_points': 5},
                'Net Profit Margin': {'high': 0.15, 'medium': 0.05, 'high_points': 10, 'medium_points': 5},
                'Return on Equity': {'high': 0.15, 'medium': 0.1, 'high_points': 5, 'medium_points': 2.5},
                'Asset Turnover': {'high': 1.0, 'medium': 0.5, 'high_points': 5, 'medium_points': 2.5},
                'Inventory Turnover': {'high': 6.0, 'medium': 4.0, 'high_points': 5, 'medium_points': 2.5},
                'Receivables Turnover': {'high': 8.0, 'medium': 4.0, 'high_points': 5, 'medium_points': 2.5}
            }
            
            # Define metric categories
            metric_categories = {
                'liquidity': ['Current Ratio', 'Quick Ratio', 'Operating Cash Flow Ratio'],
                'solvency': ['DE Ratio', 'Interest Coverage Ratio', 'DSCR'],
                'profitability': ['ROA', 'Net Profit Margin', 'Return on Equity'],
                'efficiency': ['Asset Turnover', 'Inventory Turnover', 'Receivables Turnover']
            }
            
            # Initialize scores
            scores = {category: 0 for category in metric_categories.keys()}
            
            # Calculate scores by category
            for category, metrics in metric_categories.items():
                for metric in metrics:
                    value = metrics_df[metric].iloc[0]
                    threshold = thresholds[metric]
                    
                    # Handle infinity values
                    if np.isinf(value) or np.isnan(value):
                        continue
                        
                    if 'inverted' in threshold and threshold['inverted']:
                        # For ratios where lower is better
                        if value <= threshold['low']:
                            scores[category] += threshold['low_points']
                        elif value <= threshold['medium']:
                            scores[category] += threshold['medium_points']
                    else:
                        # For ratios where higher is better
                        if value >= threshold['high']:
                            scores[category] += threshold['high_points']
                        elif value >= threshold['medium']:
                            scores[category] += threshold['medium_points']
            
            # Calculate max possible scores
            max_scores = {
                'liquidity': 30,
                'solvency': 30,
                'profitability': 25,
                'efficiency': 15
            }
            
            # Scale scores to 0-100
            total_score = sum(scores.values())
            max_total = sum(max_scores.values())
            scaled_score = min(100, (total_score / max_total) * 100) if max_total > 0 else 0
            
            return scaled_score, scores, max_scores
        
        # Calculate the health score
        overall_score, category_scores, max_scores = calculate_health_score(metrics_df)
        
        # Create columns for the scores
        score_col1, score_col2 = st.columns(2)
        with score_col1:
            # Display the overall score with a gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Financial Health Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': overall_score
                    }
                }
            ))
            
            fig.update_layout(height=400, margin=dict(t=30, b=0, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)
            
            # Health assessment text using pandas cut
            health_bins = [0, 40, 60, 80, 100]
            health_labels = ["Poor", "Fair", "Good", "Excellent"]
            health_descriptions = [
                "The company shows significant financial weaknesses that require immediate attention. There are serious concerns regarding liquidity, solvency, or profitability that could threaten viability if not addressed.",
                "The company has adequate financial health but notable areas requiring attention. There are some concerning metrics that should be addressed to improve stability and reduce financial risk.",
                "The company shows solid financial health with some areas for improvement. While fundamentally stable, there may be specific metrics that warrant attention to strengthen the overall financial position.",
                "The company demonstrates strong financial health across all areas. It has robust liquidity, manageable debt levels, and strong profitability. The company is well-positioned for growth and should have good access to capital."
            ]
            
            # Use pandas cut to assign health status and description
            health_status = pd.cut([overall_score], bins=health_bins, labels=health_labels)[0]
            health_text = pd.cut([overall_score], bins=health_bins, labels=health_descriptions)[0]
            
            st.write(f"**Status:** {health_status}")
            st.write(health_text)  

    # Decision Support Tab
    with tabs[4]:
        st.subheader("Decision Support for Bank Managers")
        
        # Credit Assessment
        st.write("### Credit Assessment")
        
        # Calculate overall credit score (simplified version)
        def calculate_credit_score(metrics_df):
            score = 0
            
            # DSCR impact (max 25 points)
            if metrics_df['DSCR'].iloc[0] >= 2.0:
                score += 25
            elif metrics_df['DSCR'].iloc[0] >= 1.5:
                score += 20
            elif metrics_df['DSCR'].iloc[0] >= 1.2:
                score += 15
            elif metrics_df['DSCR'].iloc[0] >= 1.0:
                score += 10
            
            # D/E Ratio impact (max 20 points)
            if metrics_df['DE Ratio'].iloc[0] <= 0.5:
                score += 20
            elif metrics_df['DE Ratio'].iloc[0] <= 1.0:
                score += 15
            elif metrics_df['DE Ratio'].iloc[0] <= 2.0:
                score += 10
            elif metrics_df['DE Ratio'].iloc[0] <= 3.0:
                score += 5
            
            # Interest Coverage Ratio impact (max 20 points)
            if metrics_df['Interest Coverage Ratio'].iloc[0] >= 5.0:
                score += 20
            elif metrics_df['Interest Coverage Ratio'].iloc[0] >= 3.0:
                score += 15
            elif metrics_df['Interest Coverage Ratio'].iloc[0] >= 1.5:
                score += 10
            elif metrics_df['Interest Coverage Ratio'].iloc[0] >= 1.0:
                score += 5
            
            # Current Ratio impact (max 15 points)
            if metrics_df['Current Ratio'].iloc[0] >= 2.0:
                score += 15
            elif metrics_df['Current Ratio'].iloc[0] >= 1.5:
                score += 10
            elif metrics_df['Current Ratio'].iloc[0] >= 1.0:
                score += 5
            
            # Profitability impact (max 20 points)
            if metrics_df['ROA'].iloc[0] >= 0.1:
                score += 10
            elif metrics_df['ROA'].iloc[0] >= 0.05:
                score += 5
            
            if metrics_df['Net Profit Margin'].iloc[0] >= 0.15:
                score += 10
            elif metrics_df['Net Profit Margin'].iloc[0] >= 0.05:
                score += 5
            
            return score
        
        # Calculate the credit score
        credit_score = calculate_credit_score(metrics_df)
        
        # Determine credit rating based on score
        if credit_score >= 85:
            credit_rating = "AAA"
            loan_recommendation = "Excellent candidate for loans. Minimal risk. Can offer preferential rates and flexible terms."
        elif credit_score >= 70:
            credit_rating = "AA"
            loan_recommendation = "Strong candidate for loans. Low risk. Can offer favorable rates and good terms."
        elif credit_score >= 55:
            credit_rating = "A"
            loan_recommendation = "Good candidate for loans. Moderate risk. Standard rates and terms recommended."
        elif credit_score >= 40:
            credit_rating = "BBB"
            loan_recommendation = "Acceptable candidate for loans. Moderate to higher risk. Consider shorter terms or additional collateral."
        elif credit_score >= 25:
            credit_rating = "BB"
            loan_recommendation = "Marginal candidate for loans. Higher risk. Require strong collateral, higher rates, and strict covenants."
        else:
            credit_rating = "B or below"
            loan_recommendation = "High-risk candidate. Consider declining or requiring substantial collateral and restrictive covenants."
        
        # Display credit assessment
        credit_col1, credit_col2 = st.columns([1, 2])
        
        with credit_col1:
            # Display the credit score with a gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=credit_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Credit Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 25], 'color': "red"},
                        {'range': [25, 40], 'color': "orange"},
                        {'range': [40, 55], 'color': "yellow"},
                        {'range': [55, 70], 'color': "lightgreen"},
                        {'range': [70, 85], 'color': "green"},
                        {'range': [85, 100], 'color': "darkgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': credit_score
                    }
                }
            ))
            
            fig.update_layout(height=400, margin=dict(t=30, b=0, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)
            
            # Credit rating
            st.write(f"**Credit Rating:** {credit_rating}")
            # Loan recommendation
            st.write("**Loan Recommendation:**")
            st.write(loan_recommendation)
        
        with credit_col2:
            # Key factors affecting credit decision
            st.write("### Key Factors Affecting Credit Decision")
            
            # Create a table of key factors
            factors_df = pd.DataFrame({ # Color code based on impact
                "Factor": [
                    "Debt Service Coverage Ratio",
                    "Debt-to-Equity Ratio",
                    "Interest Coverage Ratio",
                    "Current Ratio",
                    "Return on Assets"
                ],
                "Value": [
                    f"{metrics_df['DSCR'].iloc[0]:.2f}",
                    f"{metrics_df['DE Ratio'].iloc[0]:.2f}",
                    f"{metrics_df['Interest Coverage Ratio'].iloc[0]:.2f}",
                    f"{metrics_df['Current Ratio'].iloc[0]:.2f}",
                    f"{metrics_df['ROA'].iloc[0]:.2%}"
                ],
                "Impact": [
                    "High" if metrics_df['DSCR'].iloc[0] >= 1.5 else 
                    "Medium" if metrics_df['DSCR'].iloc[0] >= 1.0 else "Low",
                    
                    "High" if metrics_df['DE Ratio'].iloc[0] <= 1.0 else 
                    "Medium" if metrics_df['DE Ratio'].iloc[0] <= 2.0 else "Low",
                    
                    "High" if metrics_df['Interest Coverage Ratio'].iloc[0] >= 3.0 else 
                    "Medium" if metrics_df['Interest Coverage Ratio'].iloc[0] >= 1.5 else "Low",
                    
                    "High" if metrics_df['Current Ratio'].iloc[0] >= 2.0 else 
                    "Medium" if metrics_df['Current Ratio'].iloc[0] >= 1.0 else "Low",
                    
                    "High" if metrics_df['ROA'].iloc[0] >= 0.1 else 
                    "Medium" if metrics_df['ROA'].iloc[0] >= 0.05 else "Low"
                ]
            })
            
            # Display the dataframe
            st.dataframe(
                factors_df,
                hide_index=True,
                height=250
            )
            
            # Loan terms suggestions
            st.write("### Suggested Loan Terms")
            
            # Determine loan terms based on credit score
            if credit_score >= 70:  # AA or above
                max_ltv = "80%"
                interest_rate = "Prime + 0-1%"
                term_years = "Up to 15 years"
                collateral_req = "Standard"
                covenants = "Minimal"
            elif credit_score >= 55:  # A
                max_ltv = "75%"
                interest_rate = "Prime + 1-2%"
                term_years = "Up to 10 years"
                collateral_req = "Standard"
                covenants = "Standard financial covenants"
            elif credit_score >= 40:  # BBB
                max_ltv = "70%"
                interest_rate = "Prime + 2-3%"
                term_years = "Up to 7 years"
                collateral_req = "Additional may be required"
                covenants = "Stricter financial covenants"
            else:  # BB or below
                max_ltv = "60%"
                interest_rate = "Prime + 3-5%"
                term_years = "Up to 5 years"
                collateral_req = "Substantial required"
                covenants = "Strict covenants with regular monitoring"
            
            # Create columns for loan terms
            term_col1, term_col2 = st.columns(2)
            
            with term_col1:
                st.write(f"**Max Loan-to-Value:** {max_ltv}")
                st.write(f"**Interest Rate:** {interest_rate}")
                st.write(f"**Term Length:** {term_years}")
            
            with term_col2:
                st.write(f"**Collateral Requirements:** {collateral_req}")
                st.write(f"**Covenants:** {covenants}")
        
        # Stress Testing Section
        st.write("### Stress Testing")
        st.write("""
        Evaluate how the company would perform under various adverse scenarios.
        Adjust the parameters below to see the impact on key financial metrics.
        """)
        
        # Create columns for stress test controls
        stress_col1, stress_col2, stress_col3 = st.columns(3)
        
        with stress_col1:
            revenue_decline = st.slider("Revenue Decline (%)", 0, 50, 10, 5,help="Percentage reduction in Sales Revenue")
            cogs_increase = st.slider("COGS Increase (%)", 0, 30, 10, 5,help="Percentage increase in Cost of Goods Sold")
        
        with stress_col2:
            interest_rate_increase = st.slider("Interest Rate Increase (pp)", 0, 5, 1, 1,help="Increase in interest rate in percentage points")
            ar_decline = st.slider("Accounts Receivable Decline (%)", 0, 50, 10, 5, help="Percentage reduction in Accounts Receivable")
        
        with stress_col3:
            debt_increase = st.slider("Debt Increase (%)", 0, 50, 10, 5, help="Percentage increase in Short-term and Long-term Debt")
            cash_decline = st.slider("Cash Decline (%)", 0, 50, 10, 5, help="Percentage reduction in Cash")
        
        # Calculate stress test impact
        def calculate_stress_test(df, metrics_df, params):
            """
            Calculate stressed financial metrics based on input parameters.
            Properly handles sign conventions in financial data.
            
            Args:
                df (DataFrame): Original financial statement data
                metrics_df (DataFrame): Original calculated financial ratios and metrics
                params (dict): Stress test parameters (e.g. revenue_decline, cogs_increase)
                
            Returns:
                tuple: (stress_df, stress_metrics) - Stressed financial statement and metrics
            """
            # Create copies of the original data
            stress_df = df.copy()
            
            # original_net_income = df["Net Income"].iloc[0]
            # original_sales_revenue = df["Sales Revenue"].iloc[0]
            # original_total_assets = df["Total Assets"].iloc[0]

            # 1. Apply revenue decline (Revenue is positive, so reduce it)
            stress_df.at[0, "Sales Revenue"] = df["Sales Revenue"].iloc[0] * (1 - params["revenue_decline"] / 100)
            
            # 2. Handle COGS correctly based on its original sign
            # In financial statements, COGS is typically positive, but we subtract it from revenue
            original_cogs = df["Cost of Goods Sold"].iloc[0]
            cogs_abs = abs(original_cogs)  # Get absolute value
            
            # Apply stress to absolute value, maintaining the original sign
            if original_cogs >= 0:
                # Conventional positive COGS that gets subtracted
                stress_df.at[0, "Cost of Goods Sold"] = cogs_abs * (1 + params["cogs_increase"] / 100)
            else:
                # Negative COGS (unusual but handling it)
                stress_df.at[0, "Cost of Goods Sold"] = -cogs_abs * (1 + params["cogs_increase"] / 100)
            
            # 3. Recalculate Gross Profit correctly
            stress_df.at[0, "Gross Profit"] = stress_df["Sales Revenue"].iloc[0] - abs(stress_df["Cost of Goods Sold"].iloc[0])
            
            # 4. Handle operating expenses correctly based on their signs
            # Check if operating expenses are stored as negative (representing costs) or positive
            rd_value = df["Research & Development"].iloc[0]
            marketing_value = df["Marketing"].iloc[0]
            admin_value = df["Administrative Costs"].iloc[0]
            
            # Determine whether expenses are stored as positive or negative
            expenses_are_negative = (rd_value < 0 and marketing_value < 0 and admin_value < 0)
            
            # Calculate Operating Income correctly based on expense sign convention
            if expenses_are_negative:
                # Expenses are already negative, so add them (same as subtracting their absolute values)
                stress_df.at[0, "Operating Income"] = (
                    stress_df["Gross Profit"].iloc[0] + 
                    rd_value + 
                    marketing_value + 
                    admin_value
                )
            else:
                # Expenses are positive, so subtract them
                stress_df.at[0, "Operating Income"] = (
                    stress_df["Gross Profit"].iloc[0] - 
                    abs(rd_value) - 
                    abs(marketing_value) - 
                    abs(admin_value)
                )
            
            # 5. Interest rate increase - carefully handle Interest Expense sign
            debt_sum = df["Short-term Debt"].iloc[0] + df["Long-term Debt"].iloc[0]
            if debt_sum > 0:
                # Get original interest rate, accounting for sign
                interest_expense = df["Interest Expense"].iloc[0]
                
                if interest_expense > 0:
                    # Interest is stored as a positive number
                    original_rate = interest_expense / debt_sum
                    new_rate = original_rate * (1 + params["interest_rate_increase"] / 100)
                    stress_df.at[0, "Interest Expense"] = (stress_df["Short-term Debt"].iloc[0] + stress_df["Long-term Debt"].iloc[0]) * new_rate
                else:
                    # Interest is stored as a negative number
                    original_rate = abs(interest_expense) / debt_sum
                    new_rate = original_rate * (1 + params["interest_rate_increase"] / 100)
                    stress_df.at[0, "Interest Expense"] = -(stress_df["Short-term Debt"].iloc[0] + stress_df["Long-term Debt"].iloc[0]) * new_rate
            else:
                # No debt, so keep original interest expense
                stress_df.at[0, "Interest Expense"] = df["Interest Expense"].iloc[0]
            
            # 6. Apply debt increases
            stress_df.at[0, "Short-term Debt"] = df["Short-term Debt"].iloc[0] * (1 + params["debt_increase"] / 100)
            stress_df.at[0, "Long-term Debt"] = df["Long-term Debt"].iloc[0] * (1 + params["debt_increase"] / 100)
            
            # 7. Recalculate Income Before Tax, accounting for interest expense sign
            if df["Interest Expense"].iloc[0] > 0:
                # If interest is positive, subtract it
                stress_df.at[0, "Income Before Tax"] = stress_df["Operating Income"].iloc[0] - stress_df["Interest Expense"].iloc[0]
            else:
                # If interest is negative, add it (equivalent to subtracting a negative)
                stress_df.at[0, "Income Before Tax"] = stress_df["Operating Income"].iloc[0] + abs(stress_df["Interest Expense"].iloc[0])
            
            # 8. Recalculate Tax Expense
            if df["Income Before Tax"].iloc[0] > 0:
                tax_rate = df["Tax Expense_20"].iloc[0] / df["Income Before Tax"].iloc[0]
            else:
                tax_rate = 0.2  # Default tax rate
                
            tax_amount = stress_df["Income Before Tax"].iloc[0] * tax_rate
            stress_df.at[0, "Tax Expense_20"] = max(0, tax_amount)  # Taxes can't be negative
            
            # 9. Recalculate Net Income
            # stress_df.at[0, "Net Income"] = stress_df["Income Before Tax"].iloc[0] - stress_df["Tax Expense_20"].iloc[0]
            
            # 10. Apply AR and cash declines
            stress_df.at[0, "Accounts Receivable"] = df["Accounts Receivable"].iloc[0] * (1 - params["ar_decline"] / 100)
            stress_df.at[0, "Cash"] = df["Cash"].iloc[0] * (1 - params["cash_decline"] / 100)
            
            # 11. Calculate new metrics from stressed statement
            stress_metrics = calculate_financial_metrics(stress_df) # 2
            
            return stress_df, stress_metrics
        
        # Apply stress test
        stress_params = {
            "revenue_decline": revenue_decline,
            "cogs_increase": cogs_increase,
            "interest_rate_increase": interest_rate_increase,
            "ar_decline": ar_decline,
            "debt_increase": debt_increase,
            "cash_decline": cash_decline
        }
        
        try:
            stress_df, stress_metrics = calculate_stress_test(df, metrics_df, stress_params)
            # print("Stress DF") # print(stress_df.to_json())

            # print("Metrics DF") # print(metrics_df.to_json())
            # print(f"Original: Net Income={df['Net Income'].iloc[0]}, Total Assets={metrics_df['Total Assets'].iloc[0]}")


            # print("Stress Metrices")
            # # print(stress_metrics.to_json())
            # print(f"Stressed: Net Income={stress_df['Net Income'].iloc[0]}, Total Assets={stress_metrics['Total Assets'].iloc[0]}")
            
            # unusual relationship between Net Income and Total Assets
            # Net Income is a function of Total Assets, but the relationship is not direct.

            # Create columns for displaying the stress test results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.write("### Key Metrics After Stress Test")
                
                # raw_data = {
                #     "ROA": {
                #         "original": metrics_df['ROA'].iloc[0],
                #         "stressed": stress_metrics['ROA'].iloc[0]
                #     },
                #     "Net Profit Margin": {
                #         "original": metrics_df['Net Profit Margin'].iloc[0],
                #         "stressed": stress_metrics['Net Profit Margin'].iloc[0]
                #     }
                # }
                
                # Create comparison table
                comparison_data = {
                    "Metric": [
                        "DSCR",
                        "Interest Coverage Ratio",
                        "Current Ratio",
                        "Quick Ratio",
                        "D/E Ratio",
                        "ROA",
                        "Net Profit Margin"
                    ],
                    "Original": [
                        f"{metrics_df['DSCR'].iloc[0]:.2f}",
                        f"{metrics_df['Interest Coverage Ratio'].iloc[0]:.2f}",
                        f"{metrics_df['Current Ratio'].iloc[0]:.2f}",
                        f"{metrics_df['Quick Ratio'].iloc[0]:.2f}",
                        f"{metrics_df['DE Ratio'].iloc[0]:.2f}",
                        f"{metrics_df['ROA'].iloc[0]:.2%}",
                        f"{metrics_df['Net Profit Margin'].iloc[0]:.2%}"
                    ],
                    "Stressed": [
                        f"{stress_metrics['DSCR'].iloc[0]:.2f}",
                        f"{stress_metrics['Interest Coverage Ratio'].iloc[0]:.2f}",
                        f"{stress_metrics['Current Ratio'].iloc[0]:.2f}",
                        f"{stress_metrics['Quick Ratio'].iloc[0]:.2f}",
                        f"{stress_metrics['DE Ratio'].iloc[0]:.2f}",
                        f"{stress_metrics['ROA'].iloc[0]:.2%}",
                        f"{stress_metrics['Net Profit Margin'].iloc[0]:.2%}"
                    ]
                }
                
                print(f"Comparison Data: {comparison_data}")

                # Calculate percent change safely (avoiding division by zero)
                change_values = []
                for i, metric in enumerate(comparison_data["Metric"]):
                    # Get values and handle special formatting
                    if metric in ["ROA", "Net Profit Margin"]:
                        # For percentage values - extract from the formatted string
                        orig_val = float(comparison_data["Original"][i].strip('%')) / 100
                        stress_val = float(comparison_data["Stressed"][i].strip('%')) / 100
                    else:
                        # For regular values
                        orig_val = float(comparison_data["Original"][i])
                        stress_val = float(comparison_data["Stressed"][i])
                    
                    # Calculate percent change                                               # TODO : 10.55 Difference
                    if abs(orig_val) > 0.0001:  # Avoid division by very small numbers
                        pct_change = (stress_val - orig_val) / abs(orig_val) * 100
                    else:
                        if abs(stress_val) < 0.0001:  # Both are approximately zero          # TODO: 0.0001 is a very small number
                            pct_change = 0
                        elif stress_val > 0:
                            pct_change = float('inf')
                        else:
                            pct_change = float('-inf')
                    
                    # Format to string with % sign
                    if np.isinf(pct_change):
                        change_values.append("N/A")
                    else:
                        change_values.append(f"{pct_change:.1f}%")
                
                comparison_data["Change"] = change_values
                comparison_df = pd.DataFrame(comparison_data)
                
                print("Comparison DataFrame:")
                print(comparison_df.to_json())

                # Display the dataframe
                st.dataframe(comparison_df, hide_index=True)
            
            with result_col2:
                # Recalculate credit score based on stressed metrics
                stress_credit_score = calculate_credit_score(stress_metrics)
                
                # Create stress test visualization
                st.write("### Impact on Credit Score")
                
                # Create a gauge for the stressed credit score
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=stress_credit_score,
                    delta={"reference": credit_score, "valueformat": ".0f"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Credit Score After Stress"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'steps': [
                            {'range': [0, 25], 'color': "red"},
                            {'range': [25, 40], 'color': "orange"},
                            {'range': [40, 55], 'color': "yellow"},
                            {'range': [55, 70], 'color': "lightgreen"},
                            {'range': [70, 85], 'color': "green"},
                            {'range': [85, 100], 'color': "darkgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': stress_credit_score
                        }
                    }
                ))
                
                fig.update_layout(height=400, margin=dict(t=30, b=0, l=30, r=30))
                st.plotly_chart(fig, use_container_width=True)
                
                # Determine impact level
                score_change = stress_credit_score - credit_score
                if score_change <= -30:
                    impact_level = "Severe"
                    impact_text = "The company would face significant financial distress under these conditions."
                elif score_change <= -15:
                    impact_level = "Significant"
                    impact_text = "The company would face considerable challenges that could affect creditworthiness."
                elif score_change <= -5:
                    impact_level = "Moderate"
                    impact_text = "The company shows moderate vulnerability but should remain stable."
                else:
                    impact_level = "Minimal"
                    impact_text = "The company shows resilience to the applied stress conditions."
                
                st.write(f"**Impact Level:** {impact_level}")
                st.write(impact_text)
            
            # Sensitivity Analysis
            st.write("### Sensitivity Analysis")
            st.write("Impact of different levels of revenue decline on key metrics")
            
            # Prepare sensitivity analysis data
            sensitivity_levels = [0, 10, 20, 30, 40]
            sensitivity_data = []
            
            for level in sensitivity_levels:
                params = stress_params.copy()
                params["revenue_decline"] = level
                _, sens_metrics = calculate_stress_test(df, metrics_df, params)
                
                sensitivity_data.append({
                    "Revenue Decline": f"{level}%",
                    "DSCR": sens_metrics['DSCR'].iloc[0],
                    "Interest Coverage": sens_metrics['Interest Coverage Ratio'].iloc[0],
                    "Net Profit Margin": sens_metrics['Net Profit Margin'].iloc[0]
                })
            
            sens_df = pd.DataFrame(sensitivity_data)
            
            # Plot sensitivity chart
            sens_fig = go.Figure()
            
            # Add traces for each metric
            sens_fig.add_trace(go.Scatter(
                x=sensitivity_levels,
                y=sens_df["DSCR"],
                mode='lines+markers',
                name='DSCR'
            ))
            
            sens_fig.add_trace(go.Scatter(
                x=sensitivity_levels,
                y=sens_df["Interest Coverage"],
                mode='lines+markers',
                name='Interest Coverage'
            ))
            
            sens_fig.add_trace(go.Scatter(
                x=sensitivity_levels,
                y=sens_df["Net Profit Margin"],
                mode='lines+markers',
                name='Net Profit Margin',
                yaxis="y2"
            ))
            
            # Update layout with dual y-axis
            sens_fig.update_layout(
                xaxis=dict(title="Revenue Decline (%)"),
                yaxis=dict(
                    title="Ratio Value",
                ),
                yaxis2=dict(
                    title="Profit Margin",
                    anchor="x",
                    overlaying="y",
                    side="right",
                    tickformat=".1%"
                ),
                height=400,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(sens_fig, use_container_width=True)
            
            # Final recommendations
            st.write("### Final Recommendations")
            
            # Generate recommendations based on all analyses
            strengths = []
            weaknesses = []
            recommendations = []
            
            # Check for strengths
            if metrics_df['Current Ratio'].iloc[0] >= 2.0:
                strengths.append("Strong liquidity position with excellent current ratio")
            
            if metrics_df['DE Ratio'].iloc[0] <= 1.0:
                strengths.append("Conservative debt level with low debt-to-equity ratio")
            
            if metrics_df['Interest Coverage Ratio'].iloc[0] >= 3.0:
                strengths.append("Strong interest coverage indicating good debt service capacity")
            
            if metrics_df['Net Profit Margin'].iloc[0] >= 0.15:
                strengths.append("Excellent profit margins indicating strong pricing power or cost control")
            
            if metrics_df['ROA'].iloc[0] >= 0.1:
                strengths.append("Good return on assets indicating efficient use of resources")
            
            # Check for weaknesses
            if metrics_df['Current Ratio'].iloc[0] < 1.0:
                weaknesses.append("Poor liquidity position may lead to short-term financial stress")
            
            if metrics_df['Quick Ratio'].iloc[0] < 0.7:
                weaknesses.append("Low quick ratio indicates heavy reliance on inventory for liquidity")
            
            if metrics_df['DE Ratio'].iloc[0] > 2.0:
                weaknesses.append("High debt level increases financial risk")
            
            if metrics_df['Interest Coverage Ratio'].iloc[0] < 1.5:
                weaknesses.append("Low interest coverage ratio indicates potential debt service challenges")
            
            if metrics_df['ROA'].iloc[0] < 0.05:
                weaknesses.append("Poor return on assets indicates inefficient use of resources")
            
            # Generate recommendations
            if credit_score >= 70:
                recommendations.append("Consider offering a credit line or term loan with favorable terms")
                recommendations.append("Long-term relationship development opportunity with a financially strong client")
            elif credit_score >= 55:
                recommendations.append("Offer standard credit terms with regular financial monitoring")
                recommendations.append("Consider requiring additional financial reporting for larger credit facilities")
            elif credit_score >= 40:
                recommendations.append("Proceed with caution - require stronger collateral and more stringent covenants")
                recommendations.append("Consider shorter loan terms with more frequent review periods")
            else:
                recommendations.append("High risk client - consider declining or requiring substantial guarantees")
                recommendations.append("If proceeding, implement strict monitoring and limit exposure")
            
            # Display strengths, weaknesses, and recommendations
            sw_col1, sw_col2 = st.columns(2)
            
            with sw_col1:
                st.write("#### Strengths")
                if strengths:
                    for strength in strengths:
                        st.write(f"✅ {strength}")
                else:
                    st.write("No significant strengths identified.")
            
            with sw_col2:
                st.write("#### Areas of Concern")
                if weaknesses:
                    for weakness in weaknesses:
                        st.write(f"⚠️ {weakness}")
                else:
                    st.write("No significant concerns identified.")
            
            st.write("#### Action Items for Bank Manager")
            for i, recommendation in enumerate(recommendations, 1):
                st.write(f"{i}. {recommendation}")
                
            # Add a final summary based on the credit score
            if credit_score >= 70:
                st.write("**Summary:** This appears to be a strong credit prospect with solid financial health. Recommended to pursue business relationship with favorable terms.")
            elif credit_score >= 55:
                st.write("**Summary:** This appears to be a good credit prospect with reasonable financial health. Recommended to proceed with standard terms and appropriate monitoring.")
            elif credit_score >= 40:
                st.write("**Summary:** This credit prospect has some concerning areas, but may be acceptable with appropriate risk mitigation. Proceed with caution and enhanced monitoring.")
            else:
                st.write("**Summary:** This credit prospect presents significant risks. Consider declining or implementing substantial risk mitigation measures before proceeding.")
        
        except Exception as e:
            st.error(f"Error in stress testing: {str(e)}")
            st.write("Please check your input data and parameters for consistency.")