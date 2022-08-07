import os
import numpy as np
import pandas as pd

csv_file_path = '/home/aiffel/Code/Practice3/trade.csv'
trade = pd.read_csv(csv_file_path) 
#print(trade.head())
#print('전체 데이터 건수:', len(trade))
#print('컬럼별 결측치 개수','전체 길이 - count()')
#print(len(trade) - trade.count())
#print('기타사항은 전부 결측치 : 아무 내용이 없는 열이기 때문 고로 삭제')
trade = trade.drop('기타사항', axis=1)
#print(trade.head())
#print(trade.isnull()) # null만 보여줌
#print(trade[trade.isnull().any(axis=1)]) #null만 들어있는 Dataframe으로 보여줌
# trade에서 subset 열의 값이 모두 결측치일 때 행을 삭제 inplace 옵션으로 바로 적용
trade.dropna(how='all', subset=['수출건수', '수출금액', '수입건수', '수입금액', '무역수지'], inplace=True)
#print(trade[trade.isnull().any(axis=1)]) #null만 들어있는 Dataframe으로 보여줌
# loc[row, col] 로 해당 위치 값 가져옴
trade.loc[[188, 191, 194]] # 188, 191, 194 행 들을 가져온다.
#print(trade.loc[[188, 191, 194]])
# 191행의 비어있는 수출금액을 전/후달의 평균 금액으로 무역수지를 수출-수입으로 구해서 넣는다.
trade.loc[191,'수출금액'] = (trade.loc[188,'수출금액'] + trade.loc[194, '수출금액']) / 2
oldData = trade.loc[191,'수출금액']
trade.loc[191,'무역수지'] = (trade.loc[191,'수출금액'] - trade.loc[191, '수입금액'])
# print(trade.loc[191])
# change value by index
# trade.iloc[191, 5] = 0
# print(trade.iloc[191])

# Dulplicated data preprocessing
#print(trade[trade.duplicated()]) #only show one row(same)
# to show all rows that same values
print(trade[(trade['기간']=='2020년 03월')&(trade['국가명']=='중국')])
trade.drop_duplicates(inplace=True) # erase duplicated data and change
# print(trade[(trade['기간']=='2020년 03월')&(trade['국가명']=='중국')])
# print(trade.loc[187]) - index 187이 없어서 error
# 행 자체가 삭제된다 186, 188... 당겨주는 함수도 있음 