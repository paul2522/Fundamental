import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# 정규분포를 따라 랜덤하게 데이터 x를 생성합니다. 
x = pd.DataFrame({'A': np.random.randn(100)*4+4,
                 'B': np.random.randn(100)-1})
# print(x)
x_standardization = (x - x.mean())/x.std()
#print(x_standardization)

# 데이터 x를 min-max scaling 기법으로 정규화합니다. 
x_min_max = (x-x.min())/(x.max()-x.min())
#print(x_min_max)

fig, axs = plt.subplots(1,2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]})

axs[0].scatter(x['A'], x['B'])
axs[0].set_xlim(-5, 15)
axs[0].set_ylim(-5, 5)
axs[0].axvline(c='grey', lw=1)
axs[0].axhline(c='grey', lw=1)
axs[0].set_title('Original Data')

axs[1].scatter(x_standardization['A'], x_standardization['B'])
axs[1].set_xlim(-5, 5)
axs[1].set_ylim(-5, 5)
axs[1].axvline(c='grey', lw=1)
axs[1].axhline(c='grey', lw=1)
axs[1].set_title('Data after standardization')

plt.show()

fig, axs = plt.subplots(1,2, figsize=(12, 4),
                        gridspec_kw={'width_ratios': [2, 1]})

axs[0].scatter(x['A'], x['B'])
axs[0].set_xlim(-5, 15)
axs[0].set_ylim(-5, 5)
axs[0].axvline(c='grey', lw=1)
axs[0].axhline(c='grey', lw=1)
axs[0].set_title('Original Data')

axs[1].scatter(x_min_max['A'], x_min_max['B'])
axs[1].set_xlim(-5, 5)
axs[1].set_ylim(-5, 5)
axs[1].axvline(c='grey', lw=1)
axs[1].axhline(c='grey', lw=1)
axs[1].set_title('Data after min-max scaling')

plt.show()

cols = ['수출건수', '수출금액', '수입건수', '수입금액', '무역수지']
trade_Standardization= (trade[cols]-trade[cols].mean())/trade[cols].std()
#print(trade_Standardization.head())
#print(trade_Standardization.describe())

trade[cols] = (trade[cols]-trade[cols].min())/(trade[cols].max()-trade[cols].min())
#trade.head()
#print(trade.describe())

train = pd.DataFrame([[10, -10], [30, 10], [50, 0]])
test = pd.DataFrame([[0, 1], [10, 10]])

train_min = train.min()
train_max = train.max()
train_min_max = (train - train_min)/(train_max - train_min)
test_min_max =  (test - train_min)/(train_max - train_min)
#print(train_min_max)
#print(test_min_max)