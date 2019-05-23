# create a model to change the degree of celsius to fahrenheit
# celsius inputs : [-40, -10, 0, 8, 15, 22, 38]
# fahrenheit outputs : [-40, 14, 32, 46, 59, 72, 100]

import numpy as np


x = [-40, -10, 0, 8, 15, 22, 38]
y = [-40, 14, 32, 46, 59, 72, 100]

w = 10.0
b = 10.0

x = np.float32(x)
b = np.float32(b)


def hx(w1, w2, x_data):
    return w1 * x_data + w2


learning_rate = 0.001
'''
cost1 = hx - y
print(cost1)
cost2 = (cost1**2)**0.5
print(cost2)
cost3 = np.average(cost2)
print(cost3)
'''


# cost : mean square error
def cost(w1, w2, x_data, y_data):
    return np.average((hx(w1, w2, x_data)-y_data)**2)


def gradient(w1, w2, x_data, y_data):
    gradient_w1 = -1 * np.sum(np.average((w1*x_data+w2-y_data) * x))
    gradient_w2 = -1 * np.sum(np.average(w1*x_data+w2-y_data))
    return gradient_w1, gradient_w2


# print(cost(w, b, x, y))
# print(gradient(w, b, x, y))

for i in range(20001):
    w += gradient(w, b, x, y)[0] * learning_rate
    b += gradient(w, b, x, y)[1] * learning_rate
    # print(gradient(w, b, x, y))
    if i % 2000 == 0:
        print(w, b)

print(hx(w, b, 38))
print('The function for change degree from Celsius to Fahrenheit:'+'\n'+'F = {} x C + {}'.format(w, b))
