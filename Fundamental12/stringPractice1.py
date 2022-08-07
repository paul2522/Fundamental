my_str = 'Welcome!'
ur_str = "You're welcome."

print(my_str, ur_str)

# 아래 주석을 제거해 각 변수의 자료형을 확인해보세요 :)
print(type(my_str), type(ur_str))

print(ord('a'), ord('A'), chr(97),ord('가'))    
print(chr(0xAC00))   
#- 0xAC00은 44032의 16진수 표현입니다.

#- 파이썬 3-#
#- bytes와 string으로 구분됩니다.
str1 = b'hello'
str2 = 'hello'
str3 = u'hello'
print(type(str1), type(str2), type(str3))
# 결과 : <class 'bytes'>,  <class 'str'>,  <class 'str'>

#- 예제 코드2
print('Please don\'t touch it')
print(r'Please don\'t touch it')

EmployeeID = ['OB94382', 'OW34723', 'OB32308', 'OB83461', 'OB74830', 'OW37402', 'OW11235', 'OB82345'] 
Production_Employee = [P for P in EmployeeID if P.startswith('OB')]   # 'OB'로 시작하는 직원 ID를 다 찾아봅니다
print(Production_Employee)

import os
image_dir_path = "/home/aiffel/Code/AiffelPractice/Practice5"   
photo = os.listdir(image_dir_path )
jpeg = [jpeg for jpeg in photo if jpeg.endswith('.jpeg')]
print(jpeg)

txt = "      Strip white spaces.      "
print('[{}]'.format(txt))
print('--------------------------')

#- 양쪽 공백 제거 : strip()
print('[{}]'.format(txt.strip()))
print('--------------------------')
#- 왼쪽 공백 제거 : lstrip()
print('[{}]'.format(txt.lstrip()))
print('--------------------------')
#- 오른쪽 공백 제거 : rstrip()
print('[{}]'.format(txt.rstrip()))
#- 모든 문자를 대문자로 변환 : upper()
txt = "I fell into AIFFEL"
txt.upper()
#- 모든 문자를 소문자로 변환 : lower()
txt.lower()
#- 첫 글자만 대문자로 변환 : capitalize()
txt.capitalize()

print("aiffel".isupper())
print("aiffel".islower())
print("PYTHON".istitle())
print("python101".isalpha())
print("python101".isalnum())
print("101".isdecimal())

#- join() 사이사이 넣기
stages = ['fundamentals', 'exploration', 'goingdeeper']
"._.".join(stages)

print('fundamentals,exploration,goingdeeper'.split(','))

sent = 'I can do it!'
sent.replace('I', 'You')

sent = 'I fell into AIFFEL'
print(sent)
print(id(sent))
sent.upper()
print(sent)
print(id(sent))

sent = 'I fell into AIFFEL'
print(sent)
print(id(sent))
sent = sent.upper()
print(sent)
print(id(sent))