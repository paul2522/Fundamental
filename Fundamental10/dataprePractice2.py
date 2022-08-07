import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def outlier(df, col, z):
    return df[abs(df[col] - np.mean(df[col]))/np.std(df[col])>z].index

csv_file_path = '/home/aiffel/Code/Practice3/trade.csv'
trade = pd.read_csv(csv_file_path) 
trade = trade.drop('기타사항', axis=1)
trade.dropna(how='all', subset=['수출건수', '수출금액', '수입건수', '수입금액', '무역수지'], inplace=True)
trade.loc[[188, 191, 194]] # 188, 191, 194 행 들을 가져온다.
trade.loc[191,'수출금액'] = (trade.loc[188,'수출금액'] + trade.loc[194, '수출금액']) / 2
oldData = trade.loc[191,'수출금액']
trade.loc[191,'무역수지'] = (trade.loc[191,'수출금액'] - trade.loc[191, '수입금액'])
# print(trade[(trade['기간']=='2020년 03월')&(trade['국가명']=='중국')])
trade.drop_duplicates(inplace=True) # erase duplicated data and change

df = pd.DataFrame({'id':['001', '002', '003', '004', '002'], 
                   'name':['Park Yun', 'Kim Sung', 'Park Jin', 'Lee Han', 'Kim Min']})
print(df)
# df에서 id 열을 볼때 중복된 값을 삭제하되 마지막(뒤의)값을 남긴다.
df.drop_duplicates(subset=['id'], keep='last',inplace=True)
print(df)

# 이상치
print(trade.loc[outlier(trade, '무역수지', 1.5)])

np.random.seed(2020)
data = np.random.randn(100)  # 평균 0, 표준편차 1의 분포에서 100개의 숫자를 샘플링한 데이터 생성
data = np.concatenate((data, np.array([8, 10, -3, -5])))      # [8, 10, -3, -5])를 데이터 뒤에 추가함
data

fig, ax = plt.subplots()
ax.boxplot(data)
plt.show()

Q3, Q1 = np.percentile(data, [75 ,25])
IQR = Q3 - Q1
print(IQR)
print(data[(Q1-1.5*IQR > data)|(Q3+1.5*IQR < data)])