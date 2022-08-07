billboardchart = {
  				 1 : ["Tho Box","Roddy Ricch","2019-12-19"],
                 2 : ["Don't Start Now", "Dua Lipa", "2019-11-01"],
                 3 : ["Life Is Good", "Future Featuring Drake", "2020-02-10"],
                 4 : ["Blinding", "The Weeknd", "2019-11-29"],
                 5 : ["Circles", "Post Malone","2019-08-30"]}

with open("billboardchart.csv","w") as f:
    for i in billboardchart.values():
        data = ",".join(i)
        f.write(data+"\n")

print("슝~")

import csv

header = ["title", "singer", "released date"]

with open("billboardchart.csv","r") as inputfile:
    with open("billboardchart_out.csv","w", newline='\n') as outputfile:
        fi = csv.reader(inputfile, delimiter=',')
        fo = csv.writer(outputfile, delimiter=',')
        fo.writerow(header)
        for row in fi:
            fo.writerow(row)

print("슝~")

#- 1. 데이터를 준비합니다.
fields = ["title", "singer", "released date"]
rows = [ ["Tho Box","Roddy Ricch","2019-12-19"],
               ["Don't Start Now", "Dua Lipa", "2019-11-01"],
               ["Life Is Good", "Future Featuring Drake", "2020-02-10"],
               ["Blinding", "The Weeknd", "2019-11-29"],
               ["Circles", "Post Malone","2019-08-30"]]

print("슝~")

import pandas as pd

df=pd.DataFrame(rows, columns=fields)
df.to_csv('pandas.csv',index=False)

print("슝~")

#- 3. 동일한 내용을 csv.writer를 이용해 수행해 봅니다.
import csv 

filename = "test.csv"
with open(filename, 'w+', newline='\n') as csv_file: 
    csv_writer = csv.writer(csv_file) 
    csv_writer.writerow(fields) 
    csv_writer.writerows(rows)

print("완료")

#- test.csv 파일을 직접 열어서 눈으로 살펴 보세요. -#
df = pd.read_csv('pandas.csv')
print(df.head())