import pandas as pd
import numpy as np

df = pd.read_csv('../dados/data_treino.csv')
print('Tc statistics:')
print(df['Tc'].describe())
print(f'\nTc == 0: {(df["Tc"]==0).sum()}')
print(f'Tc > 0: {(df["Tc"]>0).sum()}')
print(f'Tc NaN: {df["Tc"].isna().sum()}')
print(f'Tc < 0: {(df["Tc"]<0).sum()}')
print(f'\nTc value counts (first 20):')
print(df['Tc'].value_counts().head(20))
print(f'\nMedian Tc (non-zero): {df[df["Tc"]>0]["Tc"].median():.2f}')
print(f'\nUnique Tc values near 0:')
print(sorted(df[df['Tc'] <= 1]['Tc'].dropna().unique())[:20])

# Check how original notebook defined classes
# Look at the original notebook
print('\n\nChecking if there is a clear binary split...')
print(f'Materials with Tc > 0 and Tc is not NaN: {df["Tc"].dropna().gt(0).sum()}')
print(f'Materials with Tc <= 0 and Tc is not NaN: {df["Tc"].dropna().le(0).sum()}')
print(f'Materials with Tc NaN: {df["Tc"].isna().sum()}')

# The dataset likely treats NaN as "not a superconductor" or uses a threshold
# Let's check if there's a natural threshold
print(f'\nPercentiles of Tc (all non-NaN):')
for p in [10, 25, 50, 75, 90]:
    print(f'  {p}th: {df["Tc"].dropna().quantile(p/100):.2f} K')
