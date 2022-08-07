import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

df = pd.DataFrame(tips)
#print(df.head())
# print(df.shape)
# print(df.describe())

# print(df['sex'].value_counts())
# print("===========================")
# print(df['time'].value_counts())
# print("===========================")
# print(df['smoker'].value_counts())
# print("===========================")
# print(df['day'].value_counts())
# print("===========================")
# print(df['size'].value_counts())
# print("===========================")

grouped = df['tip'].groupby(df['sex'])
#print(grouped.mean())
# print(grouped.size())
sex = dict(grouped.mean()) #평균 데이터를 딕셔너리 형태로 바꿔줍니다.
print(sex)
x = list(sex.keys())  
print(x)
y = list(sex.values())
print(y)

plt.bar(x = x, height = y)
plt.ylabel('tip[$]')
plt.title('Tip by Sex')
sns.barplot(data=df, x='sex', y='tip')
plt.show()

plt.figure(figsize=(10,6)) # 도화지 사이즈를 정합니다.
sns.barplot(data=df, x='sex', y='tip')
plt.ylim(0, 4) # y값의 범위를 정합니다.
plt.title('Tip by sex') # 그래프 제목을 정합니다.
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(data=df, x='day', y='tip')
plt.ylim(0, 4)
plt.title('Tip by day')
plt.show()

fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(2,2,1)
sns.barplot(data=df, x='day', y='tip',palette="ch:.25")
ax2 = fig.add_subplot(2,2,2)
sns.barplot(data=df, x='sex', y='tip')
ax3 = fig.add_subplot(2,2,4)
sns.violinplot(data=df, x='sex', y='tip')
ax4 = fig.add_subplot(2,2,3)
sns.violinplot(data=df, x='day', y='tip',palette="ch:.25")
plt.show()

sns.catplot(x="day", y="tip", jitter=False, data=tips)
plt.show()