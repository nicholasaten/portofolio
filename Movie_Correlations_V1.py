# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
from matplotlib.pyplot import figure

plt.ion()  # Interactive mode ON (opsional, bisa dihapus jika tidak perlu)

matplotlib.rcParams['figure.figsize'] = (12,8)

# Read the data
df = pd.read_csv('movies.csv')

# Look at the data
print(df.head())

# Cleaning the data
# Let's see if there is any missing data
for col in df.columns:
    percentage_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, percentage_missing))

# Drop all the missing data
df.dropna(inplace=True)

# Look the data if they were missing data
print(df.isnull().sum())

# Data types for our columns
print(df.dtypes)

# Change data types column
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')
print(df.head())

# Create correct yearcolumn
df['yearcorrect'] = df['released'].astype(str).str[:4]

# Create sort in column
print(df.sort_values(by=['gross'], inplace=False, ascending=False))

# Show all the data
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df)

# Drop any duplicates
# df.drop_duplicates()

# Scatter plot with budget vs gross
plt.scatter(x=df['budget'], y=df['gross'])

plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')

plt.show()

# Plot budget vs gross using seaborn
sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color":"red"}, line_kws={"color":"blue"})
plt.show()

# Let's start looking at correlation
df.corr()