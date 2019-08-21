"""
Integers: Recreation One

Instructions
Divisors of 42 are : 1, 2, 3, 6, 7, 14, 21, 42.
These divisors squared are: 1, 4, 9, 36, 49, 196, 441, 1764.
The sum of the squared divisors is 2500 which is 50 * 50, a square!
Given two integers m, n (1 <= m <= n)
we want to find all integers between m and n whose sum of squared divisors is itself a square.
42 is such a number.
The result will be an array of arrays or of tuples (in C an array of Pair) or a string,
each subarray having two elements,
first the number whose squared divisors is a square and then the sum of the squared divisors.

#Examples:
list_squared(1, 250) --> [[1, 1], [42, 2500], [246, 84100]]
list_squared(42, 250) --> [[42, 2500], [246, 84100]]

The form of the examples may change according to the language, see Example Tests: for more details.
Note
In Fortran - as in any other language
- the returned string is not permitted to contain any redundant trailing whitespace:
you can use dynamically allocated character strings.
"""

import numpy as np


def divisors(i):
    return np.sum(np.array([j**2 for j in range(1, i+1) if i % j == 0]))


def is_int(num):
    return num.is_integer()


def list_squared(m, n):
    # div_list = np.array([[i, divisors(i)] for i in range(m, n+1)])
    # # print(div_list)
    # # print(div_list[:, 1])
    # IterL = list(filter(is_int, np.sqrt(div_list[:, 1])))
    # return IterL
    return list([i, divisors(i)] for i in range(m, n+1) if np.sqrt(divisors(i)).is_integer())


# https://umbum.tistory.com/378
# 리스트 내장(List Comprehension)을 사용해 PC에서는 정상 수행 완료 하였으나, Codewar 홈페이지에선 실행시간이 너무 길어서 성공하지 못함.
# 효율을 개선하기 위한 시도
# 1. List Comprehension 을 generator 버전으로 바꿔준다.
# 2. map 함수나 yield를 사용(해보자, 아직 미완성)
# 오늘은 List Comprehension을 익힌 것으로 만족하자.


# print(list_squared(1, 250))

# Test.assert_equals(list_squared(1, 250), [[1, 1], [42, 2500], [246, 84100]])
# Test.assert_equals(list_squared(42, 250), [[42, 2500], [246, 84100]])
# Test.assert_equals(list_squared(250, 500), [[287, 84100]])
print(list_squared(1, 250))
print(list_squared(42, 250))
print(list_squared(250, 5000))
print(list_squared(20, 10000))
# print(list_squared(250, 50000))
