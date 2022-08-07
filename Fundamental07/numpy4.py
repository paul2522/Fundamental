from turtle import pd
import pandas as pd

ser = pd.Series([['a','b','c',3],['A','B','C',5],['aa','bb','cc',7],[10,11,12,13]])
#print(ser)
#print(ser.values)
#print(ser.index)
ser.index = ['A','B','C','D']
#print(ser.index)
# print(ser['A'])
# print(ser['C':])
ser.name = 'Alphabet'
#print(ser)

#DataFrame
d = pd.DataFrame(ser)
d.index = ['one','two','three','four']
hanguel = ['ㄱ','ㄴ','ㄷ','ㄹ']
d['Hanguel'] = hanguel
#print("\n", d)
#print(d[])
#print(d)

csv_path = "covid19_italy_region.csv"
coviddata = pd.read_csv(csv_path)
#print(coviddata)
#print(coviddata.columns)
#print(coviddata.info())

print(coviddata.describe())