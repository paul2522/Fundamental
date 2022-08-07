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

#trade 데이터의 국가명 컬럼 원본
print(trade['국가명'].head())  

# get_dummies를 통해 국가명 원-핫 인코딩
country = pd.get_dummies(trade['국가명'])
country.head()

trade = pd.concat([trade, country], axis=1)
print(trade.head())

trade.drop(['국가명'], axis=1, inplace=True)
print(trade.head())