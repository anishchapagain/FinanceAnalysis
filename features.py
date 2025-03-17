import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns

def analysis(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Set display options for better readability
    pd.set_option("display.float_format", "{:.2f}".format)

    # General Overview
    print("=== General Information ===")
    print(df.info())
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    # Statistical Summary
    print("\n=== Statistical Summary ===")
    print(df.describe())

    # Customer Segmentation
    print("\n=== Account Types Distribution ===")
    print(df["account_type"].value_counts())

    print("\n=== Industry Segmentation ===")
    print(df["customer_industry"].value_counts().head(10))

    print("\n=== Economic Sector Segmentation ===")
    print(df["economic_sector"].value_counts().head(10))

    # Balance Analysis
    print("\n=== Account Balance Statistics ===")
    print(df["account_balance"].describe())

    high_balance = df.nlargest(10, "account_balance")
    low_balance = df.nsmallest(10, "account_balance")
    print("\n=== Highest Balances ===")
    print(high_balance[["customer_name", "account_balance", "currency_code"]])
    print("\n=== Lowest Balances ===")
    print(low_balance[["customer_name", "account_balance", "currency_code"]])

    # Inactive Accounts
    inactive_accounts = df[df["account_inactive"] == True]
    print("\n=== Inactive Accounts ===")
    print(inactive_accounts[["customer_name", "last_debit_date", "last_credit_date"]].head(10))

    # Currency Distribution
    print("\n=== Currency Distribution ===")
    print(df.groupby("currency_code")["account_balance"].sum())

    # KYC Status Analysis
    print("\n=== KYC Status ===")
    print(df["kyc_status"].value_counts())

    # Visualization
    plt.figure(figsize=(10, 5))
    sns.histplot(df["account_balance"], bins=50, kde=True)
    plt.title("Distribution of Account Balances")
    plt.xlabel("Balance")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(10, 5))
    df["account_type"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Account Type Distribution")
    plt.xlabel("Account Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(10, 5))
    df.groupby("currency_code")["account_balance"].sum().plot(kind="bar", color="orange")
    plt.title("Total Balance by Currency")
    plt.xlabel("Currency")
    plt.ylabel("Total Balance")
    plt.xticks(rotation=45)
    plt.show()


def feature_engineering(df):
    """Automatically applies feature engineering to a given DataFrame"""
    df = df.copy()

    # Handling Missing Values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':  # Categorical columns
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:  # Numerical columns
                df[col].fillna(df[col].median(), inplace=True)

    # Encoding Categorical Variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    # Feature Scaling
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()

    # Generating New Features
    if "date" in df.columns or any("date" in col.lower() for col in df.columns):
        for col in df.filter(like="date").columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df.drop(columns=[col], inplace=True)

    return df

def tryit():
    df = pd.read_csv("data.csv")
    df = feature_engineering(df)
    print(df.head())
    return {"columns": df.columns.tolist(), "processed_sample": df.head(3).to_dict(orient="records")}

if __name__ == "__main__":
    data = "./data/partial_data.csv"
    print(analysis(data))
    # tryit()    