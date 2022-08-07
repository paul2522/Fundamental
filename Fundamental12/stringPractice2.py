import re

# 1단계 :  "the"라는 패턴을 컴파일한 후 패턴 객체를 리턴합니다. 
pattern = re.compile("the")    
# 2단계 : 컴파일된 패턴 객체를 활용하여 다른 텍스트에서 검색을 수행합니다.
print(pattern.findall('of the people, for the people, by the people'))
# 2단계와 같은 역할 수행
print(re.findall('the', 'of the people, for the people, by the people'))

src = "My name is..."
# 처음부터 패턴이 검색 대상과 일치할때만
regex = re.match("My", src)
print(regex)
if regex:
    print(regex.group())
else:
    print("No!")
    
#- 연도(숫자)
# 4개의 숫자 중 첫번째 숫자가 1,2
text = """
The first season of America Premiere League  was played in 1993. 
The second season was played in 1995 in South Africa. 
Last season was played in 2019 and won by Chennai Super Kings (CSK).
CSK won the title in 2000 and 2002 as well.
Mumbai Indians (MI) has also won the title 3 times in 2013, 2015 and 2017.
"""
pattern = re.compile("[1-2]\d\d\d")
pattern.findall(text)

#- 전화번호(숫자, 기호)
phonenumber = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
phone = phonenumber.search('This is my phone number 010-111-1111')
if phone:
  print(phone.group())
print('------')
phone = phonenumber.match ('This is my phone number 010-111-1111')
if phone:
  print(phone.group())
  
  # \d\d\d = d{3}
  #- 전화번호(숫자, 기호)
phonenumber = re.compile(r'\d{3}-\d{3}-\d{4}')
phone = phonenumber.search('This is my phone number 010-111-1111')
if phone:
  print(phone.group())
print('------')
phone = phonenumber.match ('This is my phone number 010-111-1111')
if phone:
  print(phone.group())
  
# \d{3}이 2번 반복 = (\d{3}-){2}
#- 전화번호(숫자, 기호)
phonenumber = re.compile(r'(\d{3}-){2}\d{4}')
phone = phonenumber.search('This is my phone number 010-111-1111')
if phone:
  print(phone.group())
print('------')
phone = phonenumber.match ('This is my phone number 010-111-1111')
if phone:
  print(phone.group())

# []안에 0-9,a-z,A-Z 사용 가능
#- 이메일(알파벳, 숫자, 기호)
text = "My e-mail adress is doingharu@aiffel.com, and tomorrow@aiffel.com"
pattern = re.compile("[0-9a-zA-Z]+@[0-9a-z]+\.[0-9a-z]+")
pattern.findall(text)