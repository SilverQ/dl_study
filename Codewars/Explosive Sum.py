"""
Explosive Sum
How many ways can you make the sum of a number?
From wikipedia: https://en.wikipedia.org/wiki/Partition_(number_theory)#
    In number theory and combinatorics, a partition of a positive integer n, also called an integer partition, is a way of writing n as a sum of positive integers. Two sums that differ only in the order of their summands are considered the same partition. If order matters, the sum becomes a composition. For example, 4 can be partitioned in five distinct ways:
    4
    3 + 1
    2 + 2
    2 + 1 + 1
    1 + 1 + 1 + 1

Examples
Basic
exp_sum(1) # 1
exp_sum(2) # 2  -> 1+1 , 2
exp_sum(3) # 3 -> 1+1+1, 1+2, 3
exp_sum(4) # 5 -> 1+1+1+1, 1+1+2, 1+3, 2+2, 4
exp_sum(5) # 7 -> 1+1+1+1+1, 1+1+1+2, 1+1+3, 1+2+2, 1+4, 5, 2+3

exp_sum(10) # 42

Explosive
exp_sum(50) # 204226
exp_sum(80) # 15796476
exp_sum(100) # 190569292
See here for more examples.
"""

# exp_sum(5) # 7 -> 1+1+1+1+1, 1+1+1+2, 1+1+3, 1+2+2, 1+4, 5, 2+3
# 5가 들어가는 경우는 1개 발생, 4가 들어가는 경우도 1개 발생, 3이 들어가는 경우는 1+1, 2의 2개 발생,
# 2가 들어가는 경우는 3개 발생하는데 2개는 3에서 발생한 경우와 중복, 1이 들어가는 경우는 5개 인데, 2개가 2가 들어가는 경우 1개가 3이 들어가는 경우
# 1 + 1 + 2 + 1 + 2 = 7이 됨
# 중복을 어떻게 제거할 것인가?
# element의 총합이 n이 되는 set의 갯수?
# element의 갯수가 1~n개로 바뀔 때 조합의 수의 합?
# 5 - 1개 : 5, 2개 : 1+4, 2+3, 3개 : 1+1+3, 1+2+2, 4개 : 1+1+1+2, 5개 : 1+1+1+1+1
# 6 - 1개 : 6, 2개 : 1+5, 2+4, 3+3, 3개 : 1+1+4, 1+2+3, 2+2+2, 4개 : 1+1+1+3, 1+1+2+2, 5개 : 1+1+1+1+2, 6개 : 1+1+1+1+1+1
# sum([0, 0, 0, 0, 0]) = 0
# ...
# sum([0, 0, 0, 0, 5]) = 5 True
# sum([1, 1, 1, 1, 1]) = 5 True
# sum([1, 1, 1, 1, 5]) = 9 False
# ...
# sum([5, 0, 0, 0, 0]) = 5 True, Duplicated
# sum([5, 5, 5, 5, 5]) = 25 False


def exp_sum(n):
    for i in range(n+1):
        result = [i]
    print(result)

# test.describe('testing exp_sum')
# test.it('***** Very basic tests *****\n')
# test.assert_equals(exp_sum(1), 1)
# test.assert_equals(exp_sum(2), 2)
# test.assert_equals(exp_sum(3), 3)
# test.it('_____ So far so good _____\n')
# test.it('\n***** Funcionality tests *****\n')
# test.assert_equals(exp_sum(4), 5)
# test.assert_equals(exp_sum(5), 7)
# test.assert_equals(exp_sum(10), 42)

# print(exp_sum(1))     # 1
# print(exp_sum(2))     # 2
print(exp_sum(3))     # 3
# print(exp_sum(4))     # 5
# print(exp_sum(5))     # 7
# print(exp_sum(10))    # 42
