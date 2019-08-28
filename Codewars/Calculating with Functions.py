"""
Calculating with Functions
Instructions
    This time we want to write calculations using functions and get the results. Let's have a look at some examples:
    seven(times(five())) # must return 35
    four(plus(nine())) # must return 13
    eight(minus(three())) # must return 5
    six(divided_by(two())) # must return 3

    Requirements:
        There must be a function for each number from 0 ("zero") to 9 ("nine")
        There must be a function for each of the following mathematical operations:
         plus, minus, times, dividedBy (divided_by in Ruby and Python)
        Each calculation consist of exactly one operation and two numbers
        The most outer function represents the left operand, the most inner function represents the right operand
        Divison should be integer division. For example, this should return 2, not 2.666666...:

    eight(divided_by(three()))
"""
import sys


def convert(word):
    dict = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9
    }
    return dict[word]


def operations(params):
    if params[1] == 'plus':
        return params[0] + params[2]
    elif params[1] == 'minus':
        return params[0] - params[2]
    elif params[1] == 'times':
        return params[0] * params[2]
    elif params[1] == 'divided_by':
        return params[0] // params[2]


def zero(num=None):
    tmp = convert(sys._getframe().f_code.co_name)
    return tmp if num is None else operations([tmp, num[0], num[1]])


def one(num=None):
    tmp = convert(sys._getframe().f_code.co_name)
    return tmp if num is None else operations([tmp, num[0], num[1]])


def two(num=None): #your code here
    tmp = convert(sys._getframe().f_code.co_name)
    return tmp if num is None else operations([tmp, num[0], num[1]])


def three(num=None): #your code here
    tmp = convert(sys._getframe().f_code.co_name)
    return tmp if num is None else operations([tmp, num[0], num[1]])


def four(num=None): #your code here
    tmp = convert(sys._getframe().f_code.co_name)
    return tmp if num is None else operations([tmp, num[0], num[1]])


def five(num=None): #your code here
    tmp = convert(sys._getframe().f_code.co_name)
    return tmp if num is None else operations([tmp, num[0], num[1]])


def six(num=None): #your code here
    tmp = convert(sys._getframe().f_code.co_name)
    return tmp if num is None else operations([tmp, num[0], num[1]])


def seven(num=None): #your code here
    tmp = convert(sys._getframe().f_code.co_name)
    return tmp if num is None else operations([tmp, num[0], num[1]])


def eight(num=None): #your code here
    tmp = convert(sys._getframe().f_code.co_name)
    return tmp if num is None else operations([tmp, num[0], num[1]])


def nine(num=None): #your code here
    tmp = convert(sys._getframe().f_code.co_name)
    return tmp if num is None else operations([tmp, num[0], num[1]])


def plus(num):
    return sys._getframe().f_code.co_name, num


def minus(num): #your code here
    return sys._getframe().f_code.co_name, num


def times(num): #your code here
    return sys._getframe().f_code.co_name, num


def divided_by(num): #your code here
    return sys._getframe().f_code.co_name, num



# Test.describe('Basic Tests')
# Test.assert_equals(seven(times(five())), 35)
# Test.assert_equals(four(plus(nine())), 13)
# Test.assert_equals(eight(minus(three())), 5)
# Test.assert_equals(six(divided_by(two())), 3)


print(seven(times(five())))        # 35
print(four(plus(nine())))          # 13
print(eight(minus(three())))       # 5
print(six(divided_by(two())))      #  3



# Other's Answers
# JordiFormadisimo, Xueyimei, g-clock
def zero(f = None): return 0 if not f else f(0)
def one(f = None): return 1 if not f else f(1)
def two(f = None): return 2 if not f else f(2)
def three(f = None): return 3 if not f else f(3)
def four(f = None): return 4 if not f else f(4)
def five(f = None): return 5 if not f else f(5)
def six(f = None): return 6 if not f else f(6)
def seven(f = None): return 7 if not f else f(7)
def eight(f = None): return 8 if not f else f(8)
def nine(f = None): return 9 if not f else f(9)

def plus(y): return lambda x: x+y
def minus(y): return lambda x: x-y
def times(y): return lambda  x: x*y
def divided_by(y): return lambda  x: x/y


# Unnamed, maplamapla
id_ = lambda x: x
number = lambda x: lambda f=id_: f(x)
zero, one, two, three, four, five, six, seven, eight, nine = map(number, range(10))
plus = lambda x: lambda y: y + x
minus = lambda x: lambda y: y - x
times = lambda x: lambda y: y * x
divided_by = lambda x: lambda y: y / x


