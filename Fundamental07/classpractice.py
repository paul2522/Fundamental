class TestClass: #()는 주로 생략
    #속성(state)
    name = "Aiffel 대구"
    start = "10:10"
    end = "18:20"
    hour = 10
    
    def __init__(self,studname = '사용자없음', studgen = '미지정'):
        self.sn = studname
        self.sg = studgen
    
    def listen(self):
        print("Aiffel 수강중")
    
    def student(self, studOrder, studNum = 0):
        self.studOrder = studOrder
        self.studNum = studNum + studOrder

# #class TestClass
# print(TestClass)        
# TestClass = TestClass()
# print(type(TestClass))
# t = TestClass()
# print(t)

    # def __init__(self,studname = '사용자없음', studgen = '미지정'):
    #     self.sn = studname
    #     self.sg = studgen

class NewClass(TestClass):          #NewClass는 TestClass를 상속받는다.
    def newlisten(self):            #add
        print("집에서도 수강!")
    
    def listen(self):               #Override
        print("Aiffel 또 수강중")
    
    def listen(self, time = 0):     #Override에 변수 추가
        self.time = time
        print("Aiffel", self.time, "번째 또 수강중")
    
    def listen(self, time = 0, classnum = 1):     #Override에 변수 추가
        self.time = time
        self.classnum = classnum
        print("Aiffel", self.classnum, "반에서", self.time, "번째 또 수강중")
        
    #오버로딩은 안되는걸로..

t = TestClass()
nC = NewClass()
#nC.newlisten()
#t.listen()
#test overloading
nC.listen()
nC.listen(10)
nC.listen(1,5)
