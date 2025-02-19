import pandas as pd

def main(file):
    df = pd.read_csv(file)
    print(df['account_type'].value_counts(normalize=True) * 100)
    branch_total = df.groupby('bank_branch').sum()
    channel_preference = df.groupby('customer_industry').apply(lambda x: {
    'mobile_banking_preference': x['mobile_banking'].mean() * 100,
    'internet_banking_preference': x['internet_banking'].mean() * 100,
    'branch_preference': (~(x['mobile_banking'] | x['internet_banking'])).mean() * 100,
    'avg_balance': x['account_balance'].mean(),
    'customer_count': len(x)
})
    # print(branch_total)
    # print(channel_preference)
    # print(df.describe())
    # print(df.info())



if __name__ == "__main__":
    main('./data/partial_data_2500.csv')
