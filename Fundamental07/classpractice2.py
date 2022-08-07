class testClass:
    pass

print(id(testClass))
print(id(testClass))    

if testClass is testClass:
    print("()없는 것은 같다")
else:
    print("()없는 것은 다르다")

class testClass():
    pass

print(id(testClass()))
print(id(testClass()))

t1 = testClass()
t2 = testClass()

print(id(t1))
print(id(t2))

if(t1 is t2):
    print("t1 t2는 같습니다.")
else:
    print("t1 t2는 다릅니다.")

print('testClass type is', type(testClass))
print('testClass() type is', type(testClass()))