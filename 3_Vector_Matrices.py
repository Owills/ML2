import numpy as np


x = np.array([[1, 2, -1],[1, 0, 1]])
y = np.array([[3, 1],[0, -1],[-2, 3]])
z = np.array([[1],[4],[6]])

q = np.array([[1, 2], [3, 5]])
p = np.array([[5],[13]])

print(x@y)

print()
print(y@x)

print()
print(z.transpose()@y)

print()
print((np.linalg.inv(q))@p)








#v = np.array([1, 5, 2, 9])
#u = np.array([3, 6, 0, -5])

# vector addition
#print("v+u = ", v+u)

# vector scaling
#print("3v = ", 3*v)

# Dot-Product
#print("u dot v = ", np.dot(u,v))
#print('Or u dot v = ', u.dot(v))

# Length / L2 Norm of a vector
#print("sqrt(v dot v) = %.2f" % np.sqrt(np.dot(v,v)))
#print("||v|| = %.2f" % np.linalg.norm(v))

#print()
#M = np.array([ [1,9,-12], [15, -2, 0] ])
#print("M = ", M.shape)
#print(M)

# matrix addition
#A = np.array([ [1, 1], [2, 1] ])
#B = np.array([ [0, 8], [7, 11] ])
#print("A+B = \n", A+B) # '\n' is the newline character

# matrix scaling
#a = 5
#print("aB = \n", a*B)

#print()
# matrix multiplicaiton
#print("shapes of A and M:", A.shape, M.shape)
#C1 = np.matmul(A, M)
#C2 = A.dot(M)
#C3 = A@M
#print("C1 = \n", C1)
#print("C2 = \n", C2)
#print("C3 = \n", C3)

# matrix transpose
#print("M^T = \n", np.transpose(M))
#print("M^T = \n", M.transpose())


# matrix inverse
#print("A^-1 = \n", np.linalg.inv(A))

#print()
# v
#print("v", v.shape, " = ", v)

# row vector v
#v = v.reshape(1,-1) # shape -1 in np.reshape means value is infered
#print("row vector v", v.shape, " = ",  v)

# column vector v
#v = v.reshape(-1,1)
#print("col vector v", v.shape, " = \n",  v)

# matrix vector multiplication
#A = np.array([[2,0],[0,1],[1,1]])
#u = np.array([1,2])
#print("u", u.shape, " = \n", u)
#print("A", A.shape, " = \n", A)

#print("Au = \n", A.dot(u))

#u = u.reshape(-1,1)
#print("u", u.shape, " = \n", u)
#print("Au = \n", A.dot(u))


# inner product as matrix multiplication
#vdotv = np.matmul(np.transpose(v), v)
#print("v dot v =", vdotv)
#print("shape of vdotv", vdotv.shape)
