f = open("hello.txt","w") 
#- open(파일명, 파일모드)
#- 파일을 열고 파일 객체를 반환합니다. 
for i in range(10):
    f.write("안녕")
    #- write() 메서드로 '안녕'을 10번 씁니다.
f.close()
#- 작업이 끝나면 close() 메서드로 닫아줍니다. *필수!
# print("완료!")

# with open("hello.txt", "r") as f:
#   print(f.read())
  
quotes = ["\n안녕하세요.\n", "반갑습니다.\n", "오랫만입니다.\n"]

with open("hello.txt", "a") as f:              # 위에서 만든 파일에 이어서 씁니다. 
    f.writelines(quotes)
    
with open('hello.txt', 'r') as f:
    hello = f.readlines()
    print(f'위치 : {f.tell()}')
    print(hello)
    print("----------------------------------------------------")
    f.seek(10)
    print(f'위치 : {f.tell()}')
    
import sys
print(sys.executable)
print(sys.path)