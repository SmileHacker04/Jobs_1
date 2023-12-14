import numpy as np
from numpy import linalg as LA
#create an array, a row vector
v = np.array([1,2,7,5])
print(v)
#[1 2 7 5]
print(v[2])
#7
#create a n=2 x d=3 matrix
A = np.array([[3,4,3],[1,6,7]])
print(A)
#[[3 4 3]
# [1 6 7]]
print(A[1,2])
#7
print(A[:, 1:3])
#[[4 3]
# [6 7]]
#adding and multiplying vectors
u = np.array([3,4,2,2])
#elementwise add
print(v+u)
#[4 6 9 7]
#elementwise multiply
print(v*u)
#[ 3 8 14 10]
# dot product
print(v.dot(u))
# 35
print(np.dot(u,v))
# 35
print(u @ v)
# 35
#matrix multiplication
B = np.array([[1,2],[6,5],[3,4]])
print(A.dot(B))
#[[36 38]
# [58 60]]
print(A @ B)
#[[36 38]
# [58 60]]
x = np.array([3,4])
print(B.dot(x))
#[11 38 25]
#norms
print(LA.norm(v))
#8.88819441732
print(LA.norm(v,1))
#15.0
print(LA.norm(v,np.inf))
#7.0
print(LA.norm(A, ’froz’))
#10.9544511501
print(LA.norm(A,2))
#10.704642743
#transpose
print(A.T)
#[[3 1]
# [4 6]
# [3 7]]
print(x.T)
#[3 4] (always prints in row format)
print(LA.matrix_rank(A))
#2
C = np.array([[1,2],[3,5]])
print(LA.inv(C))
#[[-5. 2.]
# [ 3. -1.]]
print(C @ LA.inv(C))
#[[ 1.00000000e+00 2.22044605e-16] (nearly [[1 0]
# [ 0.00000000e+00 1.00000000e+00]] [0 1]] )
