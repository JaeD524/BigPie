import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#1
df=pd.read_csv('/Users/bagjaeyong/Desktop/대학교/빅파이/kc_house_data.csv')
#2
df  #21613개 행, 21개 열
#3
df.info() # 결측치 없고 정수형 ,실수형, 마지막 열은 범주형
type(df)
#4
df.head()
#5
num_col=df.select_dtypes(include=np.number).columns.tolist()
df[num_col].describe()
df[num_col].var()
#6.
df.isna().sum()
#결측치 없음
#.7
df.nunique()
#8.
sns.countplot(x='train', data=df)
#9.
correlation = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)

#10.
df[num_col].hist()