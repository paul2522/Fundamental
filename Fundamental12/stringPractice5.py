import xml.etree.ElementTree as ET

person = ET.Element("Person")
name = ET.Element("name")
name.text = "이펠"
person.append(name)

age = ET.Element("age")
age.text = "28"
person.append(age)

ET.SubElement(person, 'place').text = '강남'
ET.dump(person)

person.attrib["id"] = "0x0001"
name.tag = "firstname"
ET.dump(person)

lastname = ET.Element('lastname', date='2020-03-20')
lastname.text = '아'
person.insert(1,lastname)
ET.dump(person)

person.remove(age)
ET.ElementTree(person).write('person.xml')

from bs4 import BeautifulSoup
import os

path = "/home/aiffel/Code/AiffelPractice/Practice5/books.xml"  # 로컬을 사용하시려면 경로는 수정해 주세요. 
with open(path, "r", encoding='utf8') as f:
    booksxml = f.read() 
    #- 파일을 문자열로 읽기
 
soup = BeautifulSoup(booksxml,'lxml') 
#- BeautifulSoup 객체 생성 : lxml parser를 이용해 데이터 분석

for title in soup.find_all('title'): 
#-  태그를 찾는 find_all 함수 이용
    print(title.get_text())