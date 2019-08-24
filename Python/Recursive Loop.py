import numpy as np


def factorial(n):
    if n == 1:
        return 1
    else:
        return n*factorial(n-1)


tot = np.floa
tot = factorial(25)

for i in range(40):
    tot = tot / 2
    print(i, tot)

# print(4194304 / 3)
# print(factorial(25) / 3**10)
