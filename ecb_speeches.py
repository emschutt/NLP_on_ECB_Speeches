import pandas as pd

df = pd.read_csv('/Users/eduardo/Downloads/all_ECB_speeches.csv', sep="|").dropna()
df