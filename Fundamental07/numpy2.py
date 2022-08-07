import numpy as np

#Cube 0~9
A = np.arange(8).reshape(2,2,2)\

print(A,end= "\n\n")
# print(B)
# print(A+B)

#print(A[:,0,:])

A = np.transpose(A , (2,1,0))

print(A)