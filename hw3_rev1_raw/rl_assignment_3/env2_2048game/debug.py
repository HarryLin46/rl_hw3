import numpy as np
a = np.random.rand(3,2)
print(a)

print(type(np.argmax(a)))
print(np.argmax(a))

# (b,c) = np.where(a==np.max(a))[0]
# print(b,c)

print(2==2.0)