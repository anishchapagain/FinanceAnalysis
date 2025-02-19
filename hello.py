import pandas as pd

def main(file):
    df = pd.read_csv(file)
    branch_total = df.groupby('bank_branch').sum()
    print(branch_total)



if __name__ == "__main__":
    main('/data/partial_data_2500.csv')
