"""
Buddy pairs
You know what divisors of a number are. The divisors of a positive integer n are said to be proper when you consider only the divisors other than n itself. In the following description, divisors will mean proper divisors. For example for 100 they are 1, 2, 4, 5, 10, 20, 25, and 50.
Let s(n) be the sum of these proper divisors of n. Call buddy two positive integers such that the sum of the proper divisors of each number is one more than the other number:
(n, m) are a pair of buddy if s(m) = n + 1 and s(n) = m + 1

For example 48 & 75 is such a pair:
    Divisors of 48 are: 1, 2, 3, 4, 6, 8, 12, 16, 24 --> sum: 76 = 75 + 1
    Divisors of 75 are: 1, 3, 5, 15, 25 --> sum: 49 = 48 + 1
Task
Given two positive integers start and limit, the function buddy(start, limit) should return the first pair (n m) of buddy pairs such that n (positive integer) is between start (inclusive) and limit (inclusive); m can be greater than limit and has to be greater than n
If there is no buddy pair satisfying the conditions, then return "Nothing" or (for Go lang) nil
Examples
(depending on the languages)
buddy(10, 50) returns [48, 75]
buddy(48, 50) returns [48, 75]
or
buddy(10, 50) returns "(48 75)"
buddy(48, 50) returns "(48 75)"
Note
    for C: The returned string will be free'd.
    See more examples in "Sample Tests:" of your language.
"""


def s_list(n):
    return [i for i in range(1, int(n/2)+1) if n % i == 0]


def s_old(n):
    return sum([i for i in range(1, int(n/2)+1) if n % i == 0]) - 1


def s(n):
    sum_s = 0
    i = 2
    while i <= int(n/i):
        if n % i == 0:
            sum_s += i + (0 if i == int(n/i) else int(n/i))
        i += 1
    return sum_s


def buddy(start, limit):
    results = 'Nothing'
    for i in range(start, limit):
        if i % 2 == 0:
            tmp1 = s(i)
            # print(i, tmp1)
            if tmp1 >= start:
                tmp2 = s(tmp1)
                # print(i, tmp1, tmp2, tmp2 == i)
                if tmp2 == i:
                    results = [i, tmp1]
                    # print(results)
                    break
    return results
# 58710에서 STDERR 에러 발생


# print(buddy(57345, 90061))
# Test.assert_equals(buddy(10, 50), [48, 75])
# Test.assert_equals(buddy(2177, 4357), "Nothing")
# Test.assert_equals(buddy(57345, 90061), [62744, 75495])
# Test.assert_equals(buddy(1071625, 1103735), [1081184, 1331967])
# print(buddy(10, 50))
# Test.assert_equals(buddy(1071625, 1103735), [1081184, 1331967])
print(buddy(10, 50))
print(buddy(2177, 4357))
print(buddy(57345, 90061))
print(buddy(1071625, 1103735))


# Run Sample Tests까지는 통과(5203ms소요)했으나, Attempt까지는 실패(Execution Timed Out)함
# 보다 빠르게 해결할 수 있는 방안을 찾아보쟈.
