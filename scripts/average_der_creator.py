import pandas as pd

input_csv = "/workspaces/ds_project/csv/data0.csv"
df = pd.read_csv(input_csv)

df['avg'] = df.iloc[:, 2:-1].mean(axis=1)
df['first_der'] = df['avg'] - df['avg'].shift(-1)
df['Spectral_Bandwidth'] = df.iloc[:, 2:6].mean(axis=1) - df.iloc[:, 12:21].mean(axis=1)

output_csv = input_csv.replace('.csv', '_avg_&_derivative.csv')
output_csv = output_csv.replace("csv/", "avg_&_derivative_csv/")
df.to_csv(output_csv, index=False)

print(df.head())
