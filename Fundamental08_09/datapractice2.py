import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

df = pd.DataFrame(tips)

plt.figure()
sns.scatterplot(data=df , x='total_bill', y='tip', palette="ch:r=-.2,d=.3_r")
plt.show()

sns.scatterplot(data=df , x='total_bill', y='tip', hue='day')
plt.show()

plt.plot(np.random.randn(50).cumsum())
plt.show()

x = np.linspace(0, 10, 100) 
plt.plot(x, np.sin(x), 'o')
plt.plot(x, np.cos(x)) 
plt.show()

sns.lineplot(x=x, y=np.sin(x))
sns.lineplot(x=x, y=np.cos(x))
plt.show()

#히스토그램
#그래프 데이터 
mu1, mu2, sigma = 100, 130, 15
x1 = mu1 + sigma*np.random.randn(10000)
x2 = mu2 + sigma*np.random.randn(10000)

# 축 그리기
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

# 그래프 그리기
patches = ax1.hist(x1, bins=50, density=False) #bins는 x값을 총 50개 구간으로 나눈다는 뜻입니다.
patches = ax1.hist(x2, bins=50, density=False, alpha=0.5)
ax1.xaxis.set_ticks_position('bottom') # x축의 눈금을 아래 표시 
ax1.yaxis.set_ticks_position('left') #y축의 눈금을 왼쪽에 표시

# 라벨, 타이틀 달기
plt.xlabel('Bins')
plt.ylabel('Number of Values in Bin')
ax1.set_title('Two Frequency Distributions')

# 보여주기
plt.show()

sns.histplot(df['total_bill'], label = "total_bill")
sns.histplot(df['tip'], label = "tip").legend()# legend()를 이용하여 label을 표시해 줍니다.
plt.show()

df['tip_pct'] = df['tip'] / df['total_bill']
df['tip_pct'].hist(bins=50)
plt.show()

df['tip_pct'].plot(kind='kde')
plt.show()